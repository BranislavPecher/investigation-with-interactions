import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import os

from helpers import compute_accuracy, tensors_to_device


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



class ProtoNetMeta(object):

    def __init__(self, model, params={}, device=None):
        self.meta_lr = params.get('meta_lr', 1e-3)
        self.model = model.to(device=device)
        self.device = device
        self.meta_optimizer = params['meta_optimizer'](self.model.parameters(), lr=self.meta_lr)
        self.lr_scheduler = params['lr_scheduler'](optimizer=self.meta_optimizer, gamma=0.1, step_size=5)


    def prepare_results_dictionary(self, batch):
        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.,  
            'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
        }

        return results, num_tasks

    
    def meta_iteration(self, dataloader, epoch, num_epochs, max_batches=500):
        num_batches = 0
        accuracy = []
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            if 'test' not in batch:
                raise RuntimeError('The batch does not contain any test dataset.')
            batch = tensors_to_device(batch, device=self.device)

            results, num_tasks = self.prepare_results_dictionary(batch)

            mean_outer_loss = torch.tensor(0., device=self.device)

            for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(
                    zip(*batch['train'], *batch['test'])):
                self.meta_optimizer.zero_grad()
                prototypes, n_classes = self.inputs_to_prototypes(train_inputs, train_targets)

                loss, acc = self.calculate_loss(prototypes, test_inputs, test_targets, n_classes)
                results['outer_losses'][task_id] = loss.item()
                mean_outer_loss += loss
                    
                results['accuracies_after'][task_id] = acc.item()
                accuracy.append(acc.item())
                loss.backward()
                self.meta_optimizer.step()
            mean_outer_loss.div_(num_tasks)
            results['mean_outer_loss'] = mean_outer_loss.item()
            yield results
            num_batches += 1
        self.lr_scheduler.step()
    
    def evaluate(self, dataloader, max_batches=500):
        validation_accuracy = []
        num_batches = 0
        while num_batches < max_batches:
            batch = dataloader.sample_data()

            batch = tensors_to_device(batch, device=self.device)

            results, num_tasks = self.prepare_results_dictionary(batch)
            mean_outer_loss = torch.tensor(0., device=self.device)

            for task_id, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(
                    zip(*batch['train'], *batch['test'])):
                prototypes, n_classes = self.inputs_to_prototypes(train_inputs, train_targets)
                loss, acc = self.calculate_loss(prototypes, test_inputs, test_targets, n_classes)

                results['outer_losses'][task_id] = loss.item()
                mean_outer_loss += loss
                    
                results['accuracies_after'][task_id] = acc.item()

                validation_accuracy.append(acc.item())

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
                prototypes, n_classes = self.inputs_to_prototypes(train_inputs, train_targets)

            with torch.set_grad_enabled(False):
                test_logits = self.predict_classes(test_inputs, prototypes, n_classes)
                _, predictions = test_logits.max(1)# torch.max(test_logits, dim=0)
            predicted.extend(predictions.cpu().detach().tolist())
            labels.extend(test_targets.cpu().detach().tolist())
        return (predicted, labels)
    
    def inputs_to_prototypes(self, inputs, targets):
        prediction = self.model(inputs)

        classes = torch.unique(targets)
        n_classes = len(classes)

        class_indices_list = [targets.eq(y).nonzero().squeeze(1) for y in range(n_classes)]
        prototypes = torch.stack([prediction[class_indices].mean(0) for class_indices in class_indices_list])
        return prototypes, n_classes


    def calculate_loss(self, prototypes, input, targets, n_classes):
        prediction = self.model(input)

        query_indices = torch.cat(list(map(lambda y: targets.eq(y).nonzero(), range(n_classes)))).view(-1)
        prediction = prediction[query_indices]

        distances = euclidean_dist(prediction, prototypes)

        n_query = int(len(targets) / n_classes)

        loss, accuracy = self.calculate_loss_(distances, n_classes, n_query)

        return loss, accuracy

    def calculate_loss_(self, distances, n_classes, n_query):
        target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        target_inds = target_inds.to(self.device)

        log_p_y = F.log_softmax(-distances, dim=1).view(n_classes, n_query, -1)

        loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)

        accuracy = y_hat.eq(target_inds.squeeze()).float().mean()

        return loss, accuracy

    def predict_classes(self, test_inputs, prototypes, n_classes):
        predictions = self.model(test_inputs)
        
        distances = euclidean_dist(predictions, prototypes)

        n_query = int(len(test_inputs) / n_classes)
        log_p_y = F.log_softmax(-distances, dim=1)

        return log_p_y

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