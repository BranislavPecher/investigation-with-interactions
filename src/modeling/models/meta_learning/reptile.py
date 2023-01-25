import copy
import torch.nn.functional as F
import torch
import numpy as np
import os

from src.modeling.models.helpers import compute_accuracy, tensors_to_device


class Reptile(object):

    def __init__(self, model, params={}, device=None):
        self.meta_lr = params['meta_lr'] if params.get('meta_lr') else 1e-3
        self.base_lr = params['base_lr'] if params.get('base_lr') else 1e-3
        self.loss_function = params['loss_function'] if params.get('loss_function') else F.cross_entropy
        self.device = device

        self.model = model.to(device=device)
        self.base_model = None
        self.meta_optimizer = params['meta_optimizer'](self.model.parameters(), lr=self.meta_lr)
        self.step_size = params['meta_lr'] if params.get('meta_lr') else 1e-3
        self.abstract_base_optimizer = params['base_optimizer']
        self.base_optimizer = None

        self.inner_iterations = params['inner_iterations'] if params.get('inner_iterations') else 3

        self.state = None

    def initiate_base_optimizer_for_network(self):
        self.base_optimizer = self.abstract_base_optimizer(self.base_model.parameters(), lr=self.base_lr)
        if self.state is not None:
            self.base_optimizer.load_state_dict(self.state)

    def train_base_model(self, data, labels):
        self.base_model.train()
        results = {'inner_losses': np.zeros((self.inner_iterations,), dtype=np.float32)}

        for step in range(self.inner_iterations):
            prediction = self.base_model(data)
            loss = self.loss_function(prediction, labels)
            results['inner_losses'][step] = loss.item()

            if (step == 0):
                results['accuracy_before'] = compute_accuracy(prediction, labels)

            self.base_optimizer.zero_grad()
            loss.backward()
            self.base_optimizer.step()

        return results

    def prepare_results_dictionary(self, batch):
        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.inner_iterations, num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,
            'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
        }

        return results, num_tasks

    def meta_iteration(self, dataloader, epoch, num_epochs, max_batches=500):
        num_batches = 0
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            if 'test' not in batch:
                raise RuntimeError('The batch does not contain any test dataset.')
            batch = tensors_to_device(batch, device=self.device)

            results, num_tasks = self.prepare_results_dictionary(batch)

            mean_outer_loss = torch.tensor(0., device=self.device)
            # self.base_model = copy.deepcopy(self.model).to(self.device)
            # self.initiate_base_optimizer_for_network()
            for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(
                    zip(*batch['train'], *batch['test'])):
                # print(f'Batch: {num_batches}, Task: {task_id}')
                self.base_model = copy.deepcopy(self.model).to(self.device)
                self.initiate_base_optimizer_for_network()

                adaptation_results = self.train_base_model(train_inputs, train_targets)

                results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

                with torch.set_grad_enabled(self.base_model.training):
                    outer_loss, outer_accuracy = self.evaluate_(test_inputs, test_targets)
                    results['outer_losses'][task_id] = outer_loss.item()
                    mean_outer_loss += outer_loss

                results['accuracies_after'][task_id] = outer_accuracy

                # for prior_param, after_param in zip(self.model.parameters(), self.base_model.parameters()):
                #    prior_param.data = prior_param.data + self.step_size * (after_param.data - prior_param.data)

                self.model.point_grad_to(self.base_model, device=self.device)
                self.meta_optimizer.step()
            # for prior_param, after_param in zip(self.model.parameters(), self.base_model.parameters()):
            #    prior_param.data = prior_param.data + self.step_size * (after_param.data - prior_param.data)

            mean_outer_loss.div_(num_tasks)
            results['mean_outer_loss'] = mean_outer_loss.item()

            yield results

            num_batches += 1

        self.step_size = self.step_size * (1 - epoch / num_epochs)

    def evaluate(self, dataloader, max_batches=500):
        num_batches = 0
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            batch = tensors_to_device(batch, device=self.device)

            results, num_tasks = self.prepare_results_dictionary(batch)
            mean_outer_loss = torch.tensor(0., device=self.device)

            for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(
                    zip(*batch['train'], *batch['test'])):
                self.base_model = copy.deepcopy(self.model).to(self.device)
                self.initiate_base_optimizer_for_network()

                adaptation_results = self.train_base_model(train_inputs, train_targets)

                results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

                with torch.set_grad_enabled(self.base_model.training):
                    outer_loss, outer_accuracy = self.evaluate_(test_inputs, test_targets)
                    results['outer_losses'][task_id] = outer_loss.item()
                    mean_outer_loss += outer_loss

                results['accuracies_after'][task_id] = outer_accuracy

            mean_outer_loss.div_(num_tasks)
            results['mean_outer_loss'] = mean_outer_loss.item()

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
                self.base_model = copy.deepcopy(self.model).to(self.device)
                self.initiate_base_optimizer_for_network()
                _ = self.train_base_model(train_inputs, train_targets)

            with torch.set_grad_enabled(self.base_model.training):
                test_logits = self.base_model(test_inputs)
                _, predictions = torch.max(test_logits, dim=1)
            predicted.extend(predictions.cpu().detach().tolist())
            labels.extend(test_targets.cpu().detach().tolist())
        return (predicted, labels)

    def evaluate_(self, data, labels):
        prediction = self.base_model(data)

        loss = self.loss_function(prediction, labels)

        accuracy = compute_accuracy(prediction, labels)

        return loss, accuracy

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

