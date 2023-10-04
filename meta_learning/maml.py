from collections import OrderedDict
import torch
import numpy as np
from torchmeta.utils import gradient_update_parameters
import torch.nn.functional as F
from helpers import compute_accuracy, tensors_to_device
import os


class ModelAgnosticMetaLearning(object):
    """Meta-learner class for Model-Agnostic Meta-Learning.
    """
    def __init__(self, model, params={}, first_order=False, device=None):
        self.model = model.to(device=device)
        self.meta_lr = params['meta_lr'] if params.get('meta_lr') else 1e-3
        self.base_lr = params['base_lr'] if params.get('base_lr') else 1e-3
        self.optimizer = params['meta_optimizer'](self.model.parameters(), lr=self.meta_lr)
        self.step_size = params['base_lr'] if params.get('base_lr') else 1e-3
        self.first_order = first_order
        self.num_adaptation_steps = params['inner_iterations'] if params.get('inner_iterations') else 1
        self.scheduler = params.get('scheduler')
        self.loss_function = params['loss_function'] if params.get('loss_function') else F.cross_entropy
        self.device = device

        self.per_param_step_size = params.get('per_param_step_size', False)
        self.learn_step_size = params.get('learn_step_size', False)

        if self.per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(self.base_lr,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=self.learn_step_size)) for (name, param)
                                         in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(self.base_lr, dtype=torch.float32,
                                          device=self.device, requires_grad=self.learn_step_size)

        if (self.optimizer is not None) and self.learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
            if self.per_param_step_size else [self.step_size]})
            if self.scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])

    def adapt(self, inputs, targets, num_adaptation_steps=1, step_size=0.1, first_order=False):
        params = None

        results = {'inner_losses': np.zeros((num_adaptation_steps,), dtype=np.float32)}

        for step in range(num_adaptation_steps):
            logits = self.model(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            if step == 0:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            params = gradient_update_parameters(self.model, inner_loss,
                                                step_size=self.step_size, params=params,
                                                # first_order=self.first_order)
                                                first_order=(not self.model.training) or self.first_order)

        return params, results

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                                      num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
        }

        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(
                zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                                                    num_adaptation_steps=self.num_adaptation_steps,
                                                    step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            results['accuracies_after'][task_id] = compute_accuracy(test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def meta_iteration(self, dataloader, epoch, num_epochs, max_batches=500):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                               'optimizer is `None`. In order to train `{0}`, you must '
                               'specify a Pytorch optimizer as the argument of `{0}` '
                               '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                               'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            if self.scheduler is not None:
                self.scheduler.step(epoch=num_batches)

            self.optimizer.zero_grad()

            batch = tensors_to_device(batch, device=self.device)
            outer_loss, results = self.get_outer_loss(batch)
            yield results

            outer_loss.backward()
            self.optimizer.step()

            num_batches += 1

    def evaluate(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            batch = tensors_to_device(batch, device=self.device)
            _, results = self.get_outer_loss(batch)
            yield results

            num_batches += 1

    def evaluate_in_batch(self, dataloader, batch):
        batch_number = 0
        self.model.eval()
        predicted = []
        labels = []
        for batch_idx, batch in enumerate(dataloader.batch_data_for_evaluation(batch)):
            batch = tensors_to_device(batch, device=self.device)
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            if batch_idx == 0:
                params, adaptation_results = self.adapt(train_inputs, train_targets,
                                                        num_adaptation_steps=self.num_adaptation_steps,
                                                        step_size=self.step_size, first_order=self.first_order)

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                _, predictions = torch.max(test_logits, dim=1)
            predicted.extend(predictions.cpu().detach().tolist())
            labels.extend(test_targets.cpu().detach().tolist())
        return (predicted, labels)
    
    def serialize(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))

    def load_serialization(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
    
    def serialize_or_load(self, path):
        if self.serialization_exists(path):
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        else:
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
    
    def serialization_exists(self, path):
        return os.path.exists(os.path.join(path, 'model.pt'))


MAML = ModelAgnosticMetaLearning
