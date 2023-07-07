import random
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset

class ICLDataset(Dataset):

    def __init__(self, train_data, test_data, num_shots=4, num_classes=2, choice_seed=None, order_seed=None, device=None):
        self.num_shots = num_shots
        self.device = device

        self.train_data, self.train_labels = train_data
        self.train_labels = np.array(self.train_labels)
        
        self.test_data, self.test_labels = test_data
        self.test_labels = np.array(self.test_labels)

        self.choice_seed = choice_seed
        self.order_seed = order_seed

        self.num_classes = num_classes

        self.samples = None
        self.labels = None
        self.label_order = None
        self.test_text_prompts = None


    def choose_samples_for_prompt(self, seed=None):
        true_labels = np.where(self.train_labels == 1)[0]
        false_labels = np.where(self.train_labels == 0)[0]

        random.seed(seed)
        random.shuffle(true_labels)
        random.shuffle(false_labels)

        shots_per_class = int(self.num_shots / self.num_classes)

        true_samples = self.train_data[true_labels[:shots_per_class]]
        false_samples = self.train_data[false_labels[:shots_per_class]]

        self.samples = true_samples + false_samples
        self.labels = shots_per_class * [1] + shots_per_class * [0]


    def reorder_samples_for_prompt(self, seed=None):
        label_indices = np.arrange(len(self.labels))

        random.seed(seed)
        random.shuffle(label_indices)

        self.label_order = label_indices

    
    def prepare_prompt_for_data(self, dataset_instruction):
        prompt = f"""### Instruction:
        {dataset_instruction} Answer using either Yes or No. The first {self.num_shots} provided pairs of Sentence/Answer serve as context. Provide answer only for the Test Sentence.

        ### Input:
        """

        for order_idx in self.label_order:
            text = self.samples[order_idx]
            label = self.labels[order_idx]

            prompt += f"""Sentence: \"{text}\"
            Answer: {'Yes' if label == 1 else 'No'}\n"""
        
        prompt += """\nTest Sentence: \"{text}\"
        ### Response: """

        texts = []
        for text in self.test_data:
            new_prompt = prompt.format(text)
            texts.append(new_prompt)

        self.test_text_prompts = texts

    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx <= len(self.test_labels):
            test_data = self.test_text_prompts[start_idx:end_idx]
            test_labels = self.test_labels[start_idx:end_idx]

            yield (test_data, test_labels)

    def iter_data_for_evaluation(self):
        for idx in range(len(self.test_labels)):
            test_data = self.test_text_prompts[idx]
            test_labels = self.test_labels[idx]

            yield test_data, test_labels
