import random
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset
from src.modeling.models.helpers import tensors_to_device

class MetaTextDataset(Dataset):
    def __init__(self, train_data, test_data, num_tasks=None, num_shots=None, max_length=50, preembed_data=False, task_definition='random', seed=None, tasks=None, device=None, transfer_learning=False):

        self.num_tasks = num_tasks
        self.num_shots = num_shots

        self.max_length = max_length

        self.preembed_data = preembed_data
        self.task_definition = task_definition
        self.device = device
        self.num_classes = 2
        self.transfer_learning = transfer_learning

        self.test_data, self.test_labels = test_data
        if preembed_data:
            print('Pre-embeding data!')
            self.test_data = [self.prepare_data(self.test_data[idx]) for idx in range(len(self.test_data))]
            if self.device:
                self.test_data = torch.stack(self.test_data)
                self.test_labels = torch.tensor(self.test_labels)
                self.test_data = tensors_to_device(self.test_data, self.device)
                self.test_labels = tensors_to_device(self.test_labels, self.device)
            else:
                self.test_data = np.array(self.test_data)


        if task_definition != 'predefined':
            self.train_data, self.train_labels = train_data
            self.size = len(self.train_data) if train_data else 0

            if preembed_data:
                print('Pre-embeding data!')
                self.train_data = [self.prepare_data(self.train_data[idx]) for idx in range(len(self.train_data))]
                if self.device:
                    self.train_data = torch.stack(self.train_data)
                    self.train_labels = torch.tensor(self.train_labels)
                    self.train_data = tensors_to_device(self.train_data, self.device)
                    self.train_labels = tensors_to_device(self.train_labels, self.device)
                else:
                    self.train_data = np.array(self.train_data)

            if task_definition == 'from_seed':
                self.seed = seed
                if not self.transfer_learning:
                    self.split_data_to_tasks()
        else:
            self.tasks = tasks
            if self.device:
                self.tasks = tensors_to_device(self.tasks, self.device)
            
        
        self.static_adaptation_data = None
        if self.task_definition != 'random':
            self.task_idx = 0
    

    def initialize_with_parameters(self, num_tasks, num_shots):
        self.num_tasks = num_tasks
        self.num_shots = num_shots

    
    def initialize_static_adaptation_data(self, data):
        adaptation_data = data[0]
        adaptation_data = [self.prepare_data(adaptation_data[idx]) for idx in range(len(adaptation_data))]
        labels = data[1]
        if self.device:
            adaptation_data = torch.stack(adaptation_data)
            labels = torch.tensor(labels)
            adaptation_data = tensors_to_device(adaptation_data, self.device)
            labels = tensors_to_device(labels)
        else:
            adaptation_data = np.array(adaptation_data)
        self.static_adaptation_data = (adaptation_data, labels)


    def reorder_static_adaptation_data(self, seed):
        data, labels = self.static_adaptation_data
        indices = list(range(labels.shape[0]))
        random.seed(seed)
        random.shuffle(indices)
        self.static_adaptation_data = (data[indices], labels[indices])


    def reset_static_adaptation_data(self):
        self.static_adaptation_data = None


    def prepare_data(self, data):
        sample = data[0, :self.max_length]
        if sample.shape[0] < self.max_length:
            sample = np.vstack((
                np.zeros((self.max_length - sample.shape[0], sample.shape[1])),
                sample
            ))
        if self.device:
            return torch.tensor(sample).float()
        else:
            return sample

    def sample_data(self):        
        # Random sampling of data
        if self.task_definition == 'random':
            return self.__sample_data_randomly__()
        else:
            return self.__sample_data_from_static__()

    
    def sample_data_in_batch(self, epoch, batch_size=16):
        indices = np.arange(self.size)
        if self.transfer_learning and self.task_definition == 'from_seed':
            np.random.seed(self.seed[epoch])
        np.random.shuffle(indices)

        start_idx = 0
        end_idx = batch_size
        while start_idx < self.size:
            yield self.train_data[indices[start_idx : end_idx]], self.train_labels[indices[start_idx : end_idx]]
            start_idx += batch_size
            end_idx += batch_size
        
        

    def __sample_data_randomly__(self):
        if not self.preembed_data:
            np_train_labels = np.array(self.train_labels)
            train_true_labels = np.where(np_train_labels == 1)[0]        
            train_false_labels = np.where(np_train_labels == 0)[0]
            del np_train_labels

            np_test_labels = np.array(self.test_labels)
            test_true_labels = np.where(np_test_labels == 1)[0]
            test_false_labels = np.where(np_test_labels == 0)[0]
            del np_test_labels
        else:
            train_true_labels = np.where(self.train_labels.cpu() == 1)[0]        
            train_false_labels = np.where(self.train_labels.cpu() == 0)[0]

            test_true_labels = np.where(self.test_labels.cpu() == 1)[0]
            test_false_labels = np.where(self.test_labels.cpu() == 0)[0]

        sampled_train_indices = np.concatenate((
            np.random.choice(train_false_labels, (self.num_tasks, self.num_shots)),
            np.random.choice(train_true_labels, (self.num_tasks, self.num_shots))
        ), axis=1)


        sampled_test_indices = np.concatenate((
            np.random.choice(test_false_labels, (self.num_tasks, self.num_shots)),
            np.random.choice(test_true_labels, (self.num_tasks, self.num_shots))
        ), axis=1)

        np.random.shuffle(sampled_train_indices.T)
        np.random.shuffle(sampled_test_indices.T)

        if not self.preembed_data:
            train_data = [[self.prepare_data(self.train_data[idx]) for idx in row_indices] for row_indices in sampled_train_indices]
            train_labels = [self.train_labels[row_indices] for row_indices in sampled_train_indices]

            test_data = [[self.prepare_data(self.test_data[idx]) for idx in row_indices] for row_indices in sampled_test_indices]
            test_labels = [self.test_labels[row_indices] for row_indices in sampled_test_indices]

            if self.device:
                return OrderedDict([
                    ('train', (torch.stack(train_data), torch.stack(train_labels))),
                    ('test', (torch.stack(test_data), torch.stack(test_labels)))
                ])
            else:
                return OrderedDict([
                    ('train', (torch.tensor(train_data).float(), torch.tensor(train_labels))),
                    ('test', (torch.tensor(test_data).float(), torch.tensor(test_labels)))
                ])
        else:
            sampled_train_indices = sampled_train_indices.flatten()
            sampled_test_indices = sampled_test_indices.flatten()
            train_data = self.train_data[sampled_train_indices].view(self.num_tasks, self.num_classes * self.num_shots, self.train_data.shape[1], self.train_data.shape[2])
            train_labels = self.train_labels[sampled_train_indices].view(self.num_tasks, self.num_classes * self.num_shots)

            test_data = self.test_data[sampled_test_indices].view(self.num_tasks, self.num_classes * self.num_shots, self.test_data.shape[1], self.test_data.shape[2])
            test_labels = self.test_labels[sampled_test_indices].view(self.num_tasks, self.num_classes * self.num_shots)

            return OrderedDict([
                ('train', (train_data, train_labels)),
                ('test', (test_data, test_labels))
            ])

    def __sample_data_from_static__(self):
        train_tasks_data = []
        test_tasks_data = []
        train_tasks_labels = []
        test_tasks_labels = []
        for task_number in range(self.num_tasks):
            train_data, train_labels = self.tasks[self.task_idx]['train']
            test_data, test_labels = self.tasks[self.task_idx]['test']
            self.increase_task_index()

            train_tasks_data.append(train_data)
            train_tasks_labels.append(train_labels)
            test_tasks_data.append(test_data)
            test_tasks_labels.append(test_labels)

        if self.device:
            return OrderedDict([
                ('train', (torch.stack(train_tasks_data), torch.stack(train_tasks_labels))),
                ('test', (torch.stack(test_tasks_data), torch.stack(test_tasks_labels)))
            ])
        else:
            return OrderedDict([
                ('train', (torch.tensor(train_tasks_data).float(), torch.tensor(train_tasks_labels))),
                ('test', (torch.tensor(test_tasks_data).float(), torch.tensor(test_tasks_labels)))
            ])

    def batch_data_for_evaluation(self, batch=64):
        if not self.transfer_learning:
            if self.task_definition == 'random':
                if self.static_adaptation_data is None:
                    train_true_labels = [idx for idx, label in enumerate(self.train_labels) if label == 1]
                    train_false_labels = [idx for idx, label in enumerate(self.train_labels) if label == 0]

                    train_indices = random.sample(train_true_labels, self.num_shots) + random.sample(train_false_labels, self.num_shots)
                    random.shuffle(train_indices)
                    if self.preembed_data:
                        train_data = self.train_data[train_indices]
                    else:
                        train_data = [self.prepare_data(self.train_data[idx]) for idx in train_indices]
                    train_labels = self.train_labels[train_indices]
                else:
                    train_data, train_labels = self.static_adaptation_data
            else:
                index = random.sample(range(len(self.tasks)), 1)[0]
                train_data, train_labels = self.tasks[index]['train']

        test_indices = list(range(len(self.test_labels)))
        np.random.shuffle(test_indices)
        start_idx = 0
        end_idx = batch
        while start_idx <= len(test_indices):
            if self.preembed_data:
                test_data = self.test_data[test_indices[start_idx:end_idx]]
            else:
                test_data = [self.prepare_data(self.test_data[idx]) for idx in test_indices[start_idx:end_idx]]
            test_labels = self.test_labels[test_indices[start_idx:end_idx]]

            if self.transfer_learning:
                yield OrderedDict([
                    ('test', (torch.tensor(test_data).float(), torch.tensor(test_labels)))
                ])
            else:
                yield OrderedDict([
                    ('train', (torch.tensor(train_data).float(), torch.tensor(train_labels))),
                    ('test', (torch.tensor(test_data).float(), torch.tensor(test_labels)))
                ])
            start_idx = end_idx
            end_idx = end_idx + batch

    def reset_task_index(self):
        self.task_idx = 0


    def increase_task_index(self):
        self.task_idx = (self.task_idx + 1) % len(self.tasks)

    def split_data_to_tasks(self):
        self.tasks = []

        train_true_indices = [idx for idx, label in enumerate(self.train_labels) if label == 1]
        train_false_indices = [idx for idx, label in enumerate(self.train_labels) if label == 0]

        test_true_indices = [idx for idx, label in enumerate(self.test_labels) if label == 1]
        test_false_indices = [idx for idx, label in enumerate(self.test_labels) if label == 0]

        random.seed(self.seed)
        random.shuffle(train_true_indices)

        random.seed(self.seed)
        random.shuffle(train_false_indices)

        random.seed(self.seed)
        random.shuffle(test_true_indices)

        random.seed(self.seed)
        random.shuffle(test_false_indices)

        stop = False
        train_true_index = 0
        train_false_index = 0
        test_true_index = 0
        test_false_index = 0
        used_all_true = False
        used_all_false = False
        while not stop:
            train_data = []
            train_labels = []
            for _ in range(self.num_shots):
                train_data.append(self.train_data[train_true_indices[train_true_index]])
                train_labels.append(self.train_labels[train_true_indices[train_true_index]])
                train_data.append(self.train_data[train_false_indices[train_false_index]])
                train_labels.append(self.train_labels[train_false_indices[train_false_index]])

                train_true_index += 1
                if train_true_index >= len(train_true_indices):
                    train_true_index = train_true_index % len(train_true_indices)
                    used_all_true = True

                train_false_index += 1
                if train_false_index >= len(train_false_indices):
                    train_false_index = train_false_index % len(train_false_indices)
                    used_all_false = True
            
            test_data = []
            test_labels = []
            for _ in range(self.num_shots):
                test_data.append(self.test_data[test_true_indices[test_true_index]])
                test_labels.append(self.test_labels[test_true_indices[test_true_index]])
                test_data.append(self.test_data[test_false_indices[test_false_index]])
                test_labels.append(self.test_labels[test_false_indices[test_false_index]])

                test_true_index = (test_true_index + 1) % len(test_true_indices)
                test_false_index = (test_false_index + 1) % len(test_false_indices)
            
            if self.preembed_data:
                self.tasks.append({
                    'train': (
                        np.array(train_data),
                        np.array(train_labels)
                    ),
                    'test': (
                        np.array(test_data),
                        np.array(test_labels)
                    )
                })
            else:
                self.tasks.append({
                    'train': (
                        train_data,
                        train_labels
                    ),
                    'test': (
                        test_data,
                        test_labels
                    )
                })

            if used_all_true and used_all_false:
                stop = True
    
    def embed_data_for_use(self):
        if self.task_definition == 'from_seed':
            tasks = []
            for task in self.tasks:
                train_data, train_labels = task['train']
                train_data = [self.prepare_data(X) for X in train_data]
                test_data, test_labels = task['test']
                test_data = [self.prepare_data(X) for X in test_data]
                tasks.append({
                    'train': (
                        torch.stack(train_data) if self.device else np.array(train_data),
                        torch.tensor(train_labels) if self.device else np.array(train_labels)
                    ),
                    'test': (
                        torch.stack(test_data) if self.device else np.array(test_data),
                        torch.tensor(test_labels) if self.device else np.array(test_labels)
                    )
                })
            self.tasks = tensors_to_device(tasks, self.device) if self.device else tasks
