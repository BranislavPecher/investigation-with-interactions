import torch
import copy
import os
import pickle
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from torch.utils.data import RandomSampler, DataLoader, Dataset
from transformers import BertModel, BertTokenizer

class SeededRandomSampler(RandomSampler):

    def __init__(self, dataset, replacement=False, num_samples=None, seed=0):
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
        super(SeededRandomSampler, self).__init__(dataset, replacement, num_samples)
        self.dataset = dataset

    def __iter__(self):
        size = len(self.dataset)

        old_state = torch.get_rng_state()
        torch.set_rng_state(self.state)

        if self.replacement:
            iterator = iter(torch.randint(high=size, size=(self.num_samples,), dtype=torch.int64).tolist())
        else:
            iterator = iter(torch.randperm(size).tolist())

        self.state = torch.get_rng_state()
        torch.set_rng_state(old_state)
        return iterator


class DatasetLoader():

    def __init__(self, name, batch_size, dataset, shuffle_train_seed=0):
        self.name = name
        self.batch_size = batch_size
        self.shuffle_train_seed = shuffle_train_seed
        self.train_dataset = copy.deepcopy(dataset)
        self.train_dataset.train = True

        self.test_dataset = copy.deepcopy(dataset)
        self.test_dataset.train = False

    def trainloader(self):
        sampler = SeededRandomSampler(self.train_dataset, seed=self.shuffle_train_seed)
        trainloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        return trainloader

    def testloader(self):
        testloader = DataLoader(self.test_dataset, batch_size = 64, shuffle=False, pin_memory=True)
        return testloader


class TextDataset(Dataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True):
        self.dataset_name = dataset_name
        self.train = True
        self.split_seed = split_seed
        self.label_seed = label_seed
        self.full_test = full_test

        self.train_size = train_size
        self.num_labelled = num_labelled
        self.num_labelled_test = num_labelled_test
        if not self.full_test and self.num_labelled_test == 0:
            self.num_labelled_test = self.num_labelled

        self.device = device

        self.text, self.targets = self.initialise_dataset_from_huggingface()
        self.split_train_test()
        self.select_labelled_data()


    def initialise_dataset_from_huggingface(self):
        if self.dataset_name in ('sst2', 'cola'):
            print(f'Using sst2 or cola')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            return data.sentence.tolist(), data.label.tolist()
        elif self.dataset_name == 'mrpc':
            print('Using mrpc')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation']), pd.DataFrame(dataset['test'])])
            texts = [f'Sentence 1: {sent1}; Sentence 2: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            return texts, data.label.tolist()
        elif self.dataset_name == 'rte':
            print('Using rte')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Premise: {sent1}; Hypothesis: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            return texts, data.label.tolist()
        elif self.dataset_name == 'boolq':
            print('Using BoolQ dataset')
            dataset = load_dataset('super_glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Question: {question}\nPassage: {passage}' for question, passage in zip(data.question.tolist(), data.passage.tolist())]
            return texts, data.label.tolist()
        else:
            raise NotImplemented('The dataset cannot be initiated!')


    def split_train_test(self, train_test_indices=None):
        if train_test_indices is None:
            size = len(self.text)

            old_state = torch.get_rng_state()
            torch.manual_seed(self.split_seed)

            indices = torch.randperm(size)
            split = int(self.train_size * size)
            self.train_indices = indices[:split]
            self.test_indices = indices[split:]
            torch.set_rng_state(old_state)
        else:
            self.train_indices, self.test_indices = train_test_indices

        self.train_text = [self.text[idx] for idx in self.train_indices]
        self.train_targets = [self.targets[idx] for idx in self.train_indices]

        self.test_text = [self.text[idx] for idx in self.test_indices]
        self.test_targets = [self.targets[idx] for idx in self.test_indices] 


    def select_labelled_data(self):
        if self.num_labelled > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            indices = torch.randperm(len(self.train_targets))
            to_select = int(self.num_labelled / 2)

            texts = []
            targets = []
            true_indices = []
            false_indices = []
            used_true_indices = []
            used_false_indices = []
            all_true = all_false = False
            num_true = num_false = 0
            rank = 0
            for idx in indices:
                if all_true and all_false:
                    break

                if self.train_targets[idx] == 1:
                    if not all_true:
                        texts.append(self.train_text[idx])
                        targets.append(self.train_targets[idx])
                        true_indices.append(self.train_indices[idx])
                        used_true_indices.append(rank)
                        rank += 1
                        num_true += 1
                        if num_true >= to_select:
                            all_true = True
                
                elif self.train_targets[idx] == 0:
                    if not all_false:
                        texts.append(self.train_text[idx])
                        targets.append(self.train_targets[idx])
                        false_indices.append(self.train_indices[idx])
                        used_false_indices.append(rank)
                        rank += 1
                        num_false += 1
                        if num_false >= to_select:
                            all_false = True
            self.train_text = texts
            self.train_targets = targets
            self.true_indices = true_indices
            self.false_indices = false_indices
            self.used_true_indices = used_true_indices
            self.used_false_indices = used_false_indices

            print(f'Number of selected Train samples: {len(self.train_targets)}')

            torch.set_rng_state(old_state)
        
        if not self.full_test and self.num_labelled_test > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            indices = torch.randperm(len(self.test_targets))
            to_select = int(self.num_labelled_test / 2)

            texts = []
            targets = []
            test_indices = []
            all_true = all_false = False
            num_true = num_false = 0
            for idx in indices:
                if all_true and all_false:
                    break

                if self.test_targets[idx] == 1:
                    if not all_true:
                        texts.append(self.test_text[idx])
                        targets.append(self.test_targets[idx])
                        test_indices.append(self.test_indices[idx])
                        num_true += 1
                        if num_true >= to_select:
                            all_true = True
                
                elif self.test_targets[idx] == 0:
                    if not all_false:
                        texts.append(self.test_text[idx])
                        targets.append(self.test_targets[idx])
                        test_indices.append(self.test_indices[idx])
                        num_false += 1
                        if num_false >= to_select:
                            all_false = True
            self.test_text = texts
            self.test_targets = targets
            self.test_indices = test_indices

            print(f'Number of selected Test samples: {len(self.test_targets)}')

            torch.set_rng_state(old_state)

    



class ICLDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, num_shots=4, num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5'):
        super(ICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)

        self.num_shots = num_shots
        self.num_classes = num_classes
        
        self.choice_seed = choice_seed
        self.order_seed = order_seed

        self.model_name = model_name
        self.prompts, self.targets = self.prepare_dataset_for_use()


    def prepare_dataset_for_use(self):
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        prompts, targets = self.__prepare_prompt(texts, targets, self.test_text, self.test_targets)
        return prompts, targets

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch


    def __len__(self):
        return len(self.prompts)
    

    def __choose_shots(self):
        to_choose = int(self.num_shots / self.num_classes)
        true_indices = [idx for idx, target in enumerate(self.train_targets) if target == 1]
        false_indices = [idx for idx, target in enumerate(self.train_targets) if target == 0]

        true_texts = [self.train_text[idx] for idx in true_indices]
        true_targets = [self.train_targets[idx] for idx in true_indices]

        false_texts = [self.train_text[idx] for idx in false_indices]
        false_targets = [self.train_targets[idx] for idx in false_indices]

        texts = []
        targets = []
        old_state = torch.get_rng_state()
        torch.manual_seed(self.choice_seed)

        indices = torch.randperm(len(true_texts))
        indices = indices[:to_choose]
        texts += [true_texts[idx] for idx in indices]
        targets += [true_targets[idx] for idx in indices]

        indices = torch.randperm(len(false_texts))
        indices = indices[:to_choose]
        texts += [false_texts[idx] for idx in indices]
        targets += [false_targets[idx] for idx in indices]

        torch.set_rng_state(old_state)

        return texts, targets

    def __sample_reorder(self, texts, targets):
        old_state = torch.get_rng_state()
        torch.manual_seed(self.order_seed)

        indices = torch.randperm(len(texts))
        texts = [texts[idx] for idx in indices]
        targets = [targets[idx] for idx in indices]

        torch.set_rng_state(old_state)

        return texts, targets

    def __prepare_prompt(self, texts, targets, test_texts, test_targets):
        instruction, sentence_start, answer_start = self.__prepare_dataset_keywords()
        prompt = instruction
        for idx in range(len(texts)):
            prompt += f'{sentence_start}{texts[idx].replace("{", "").replace("}", "")}\n{answer_start}: {"Yes" if targets[idx] == 1 else "No"}\n'
        prompt += '{sentence_start}{sample}\n{answer_start}:'
        
        prompts = []
        for text in test_texts:
            try:
                new_prompt = prompt.format(sentence_start=sentence_start, sample=text.replace('{', '').replace('}', ''), answer_start=answer_start)
            except:
                print(text)
                print(prompt)
                print(text.replace('{', '').replace('}', ''))
                raise
            prompts.append(new_prompt)
        return prompts, test_targets

    def __prepare_dataset_keywords(self):
        if self.dataset_name == 'sst2':
            instruction = 'Determine whether the Sentence has positive sentiment. Answer using either Yes or No.\n'
            answer_start = 'Answer'
            sentence_start = 'Sentence: '
        elif self.dataset_name == 'cola':
            instruction = 'Determine whether the Sentence is a grammatical English sentence. Answer using either Yes or No.\n'
            sentence_start = 'Sentence: '
            answer_start = 'Answer'
        elif self.dataset_name == 'mrpc':
            instruction = 'Determine whether the Sentence Pair is semantically equivalent. Answer using either Yes or No.\n'
            sentence_start = 'Sentence Pair: '
            answer_start = 'Answer'
        elif self.dataset_name == 'rte':
            instruction = 'Determine whether the Premise entails the Hypothesis. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'
        elif self.dataset_name == 'boolq':
            instruction = 'Determine whether the Passage contains Answer to the Question. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'

        return instruction, sentence_start, answer_start


class SimilarityICLDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, num_shots=4, num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5'):
        super(SimilarityICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)

        self.num_shots = num_shots
        self.num_classes = num_classes
        
        self.choice_seed = choice_seed
        self.order_seed = order_seed

        with open(os.path.join('data', f'{dataset_name}_embeddings.pkl'), 'rb') as file:
            self.embeddings = pickle.load(file)

        self.model_name = model_name
        self.prompts, self.targets = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        prompts, targets = self.__prepare_prompt(texts, targets, self.test_text, self.test_targets)
        return prompts, targets

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch
    

    def __choose_shots(self):
        to_choose = int(self.num_shots / self.num_classes)

        test_embeddings = self.embeddings[self.test_indices]
        train_true_embeddings = self.embeddings[self.true_indices]
        train_false_embeddings = self.embeddings[self.false_indices]

        texts = []
        targets = []
        old_state = torch.get_rng_state()
        torch.manual_seed(self.choice_seed)
        indices_true = cosine_similarity(test_embeddings, train_true_embeddings).argsort()[::-1][:, :to_choose]
        indices_false = cosine_similarity(test_embeddings, train_false_embeddings).argsort()[::-1][:, :to_choose]
        for true_idx, false_idx in zip(indices_true, indices_false):
            temp_texts = [self.train_text[self.used_true_indices[idx]] for idx in true_idx]
            temp_texts.extend([self.train_text[self.used_false_indices[idx]] for idx in false_idx])
            texts.append(temp_texts)
        targets = [self.train_targets[self.used_true_indices[idx]] for idx in indices_true[0]]
        targets.extend([self.train_targets[self.used_false_indices[idx]] for idx in indices_false[0]])
        torch.set_rng_state(old_state)

        return texts, targets

    def __sample_reorder(self, texts, targets):
        old_state = torch.get_rng_state()
        torch.manual_seed(self.order_seed)

        indices = torch.randperm(len(targets))
        targets = [targets[idx] for idx in indices]
        texts = [[texts_row[idx] for idx in indices] for texts_row in texts]

        torch.set_rng_state(old_state)

        return texts, targets

    def __prepare_prompt(self, texts, targets, test_texts, test_targets):
        instruction, sentence_start, answer_start = self.__prepare_dataset_keywords()
        prompts = []
        for idx, text in enumerate(test_texts):
            prompt = instruction
            for train_idx, train_text in enumerate(texts[idx]):
                prompt += f'{sentence_start}{train_text.replace("{", "").replace("}", "")}\n{answer_start}: {"Yes" if targets[train_idx] == 1 else "No"}\n'
            prompt += f'{sentence_start}{text.replace("{", "").replace("}", "")}\n{answer_start}:'
            prompts.append(prompt)
        return prompts, test_targets

    def __prepare_dataset_keywords(self):
        if self.dataset_name == 'sst2':
            instruction = 'Determine whether the Sentence has positive sentiment. Answer using either Yes or No.\n'
            answer_start = 'Answer'
            sentence_start = 'Sentence: '
        elif self.dataset_name == 'cola':
            instruction = 'Determine whether the Sentence is a grammatical English sentence. Answer using either Yes or No.\n'
            sentence_start = 'Sentence: '
            answer_start = 'Answer'
        elif self.dataset_name == 'mrpc':
            instruction = 'Determine whether the Sentence Pair is semantically equivalent. Answer using either Yes or No.\n'
            sentence_start = 'Sentence Pair: '
            answer_start = 'Answer'
        elif self.dataset_name == 'rte':
            instruction = 'Determine whether the Premise entails the Hypothesis. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'
        elif self.dataset_name == 'boolq':
            instruction = 'Determine whether the Passage contains Answer to the Question. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'

        return instruction, sentence_start, answer_start


class PromptDataset(TextDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, model_name='flan-t5'):
        super(PromptDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)
        self.model_name = model_name

        self.prompts, self.targets = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        prompts, targets = self.__prepare_prompt(self.test_text, self.test_targets)
        return prompts, targets

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch
    
    def __len__(self):
        return len(self.prompts)

    def __prepare_prompt(self, test_texts, test_targets):
        instruction, sentence_start, answer_start = self.__prepare_dataset_keywords()
        prompt = instruction
        prompt += '{sentence_start}{sample}\n{answer_start}:'
        
        prompts = []
        for text in test_texts:
            new_prompt = prompt.format(sentence_start=sentence_start, sample=text.replace('{', '').replace('}', ''), answer_start=answer_start)
            prompts.append(new_prompt)
        return prompts, test_targets

    def __prepare_dataset_keywords(self):
        if self.dataset_name == 'sst2':
            instruction = 'Determine whether the Sentence has positive sentiment. Answer using either Yes or No.\n'
            answer_start = 'Answer'
            sentence_start = 'Sentence: '
        elif self.dataset_name == 'cola':
            instruction = 'Determine whether the Sentence is a grammatical English sentence. Answer using either Yes or No.\n'
            sentence_start = 'Sentence: '
            answer_start = 'Answer'
        elif self.dataset_name == 'mrpc':
            instruction = 'Determine whether the Sentence Pair is semantically equivalent. Answer using either Yes or No.\n'
            sentence_start = 'Sentence Pair: '
            answer_start = 'Answer'
        elif self.dataset_name == 'rte':
            instruction = 'Determine whether the Premise entails the Hypothesis. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'
        elif self.dataset_name == 'boolq':
            instruction = 'Determine whether the Passage contains Answer to the Question. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'

        return instruction, sentence_start, answer_start

class InstructionTuningDataset(TextDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, model_name='flan-t5'):
        super(InstructionTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)
        self.model_name = model_name

        self.prompts, self.targets = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        prompts, targets = self.__prepare_prompt(self.train_text, self.train_targets)
        return prompts, targets

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.prompts):
            data = self.prompts[start_idx : end_idx]
            labels = self.targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch
    
    def __len__(self):
        return len(self.prompts)

    def __prepare_prompt(self, texts, targets):
        instruction, sentence_start, answer_start = self.__prepare_dataset_keywords()
        prompt = instruction
        prompt += '{sentence_start}{sample}\n{answer_start}: {answer}'
        
        prompts = []
        for idx, text in enumerate(texts):
            new_prompt = prompt.format(sentence_start=sentence_start, sample=text.replace('{', '').replace('}', ''), answer_start=answer_start, answer= 'Yes' if targets[idx] == 1 else 'No')
            prompts.append(new_prompt)
        targets = ['Yes' if target == 1 else 'No' for target in targets]
        return prompts, targets

    def __prepare_dataset_keywords(self):
        if self.dataset_name == 'sst2':
            instruction = 'Determine whether the Sentence has positive sentiment. Answer using either Yes or No.\n'
            answer_start = 'Answer'
            sentence_start = 'Sentence: '
        elif self.dataset_name == 'cola':
            instruction = 'Determine whether the Sentence is a grammatical English sentence. Answer using either Yes or No.\n'
            sentence_start = 'Sentence: '
            answer_start = 'Answer'
        elif self.dataset_name == 'mrpc':
            instruction = 'Determine whether the Sentence Pair is semantically equivalent. Answer using either Yes or No.\n'
            sentence_start = 'Sentence Pair: '
            answer_start = 'Answer'
        elif self.dataset_name == 'rte':
            instruction = 'Determine whether the Premise entails the Hypothesis. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'
        elif self.dataset_name == 'boolq':
            instruction = 'Determine whether the Passage contains Answer to the Question. Answer using either Yes or No.\n'
            sentence_start = ''
            answer_start = 'Answer'

        return instruction, sentence_start, answer_start


class FineTuningDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, tokenizer=None, max_len=50):
        super(FineTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)
        self.tokenizer = tokenizer
        self.train = True
        self.max_len = max_len

        self.n_classes = len(set(self.train_targets))

    def __len__(self):
        return len(self.train_text) if self.train else len(self.test_text)

    def __getitem__(self, index):
        text = str(self.train_text[index] if self.train else self.test_text[index])
        target = self.train_targets[index] if self.train else self.test_targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class MetaLearningDataset(TextDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, max_len=50, num_tasks=16, num_shots=5, choice_seed=0, order_seed=0):
    # def __init__(self, train_data, test_data, num_tasks=None, num_shots=None, max_length=50, preembed_data=False, task_definition='random', seed=None, tasks=None, device=None, transfer_learning=False):
        super(MetaLearningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)

        self.num_tasks = num_tasks
        self.num_shots = num_shots

        self.train = True
        # self.max_len = max_len
        self.max_length = max_len
        self.n_classes = len(set(self.train_targets))

        self.device = device

        old_state = torch.get_rng_state()
        torch.manual_seed(choice_seed)
        self.choice_state = torch.get_rng_state()

        torch.manual_seed(order_seed)
        self.order_state = torch.get_rng_state()

        torch.set_rng_state(old_state)

        self.split_train_valid()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_model = BertModel.from_pretrained('bert-base-uncased')

        self.embedding_model.to(device)
        self.embedding_model.eval()

        self.train_data = self.prepare_data(self.train_text)
        self.valid_data = self.prepare_data(self.valid_text)
        self.test_targets = np.array(self.test_targets)

    def split_train_valid(self):
        size = len(self.train_text)

        old_state = torch.get_rng_state()
        torch.manual_seed(self.split_seed)

        indices = torch.randperm(size)
        split = int(self.train_size * size)
        self.train_indices = indices[:split]
        self.test_indices = indices[split:]
        torch.set_rng_state(old_state)

        self.train_text = [self.train_text[idx] for idx in self.train_indices]
        self.train_targets = np.array([self.train_targets[idx] for idx in self.train_indices])

        self.valid_text = [self.train_text[idx] for idx in self.test_indices]
        self.valid_targets = np.array([self.train_targets[idx] for idx in self.test_indices] )

    def prepare_data(self, data):
        inputs = self.tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']

        with torch.no_grad():
            embeddings = self.embedding_model(ids.to(self.device)).cpu().detach().numpy()
        return embeddings

    def sample_data(self):        
        # Random sampling of data
        self.train_targets = np.array(self.train_targets)
        self.valid_targets = np.array(self.valid_targets)

        train_true_indices = [idx for idx, target in enumerate(self.train_targets) if target == 1]
        train_false_indices = [idx for idx, target in enumerate(self.train_targets) if target == 0]

        valid_true_indices = [idx for idx, target in enumerate(self.valid_targets) if target == 1]
        valid_false_indices = [idx for idx, target in enumerate(self.valid_targets) if target == 0]
        old_state = torch.get_rng_state()
        torch.set_rng_state(self.order_state)

        train_data = []
        train_labels = []
        valid_data = []
        valid_labels = []
        for task_number in range(self.num_tasks):

            train_true_indices = torch.randperm(len(train_true_indices))
            valid_true_indices = torch.randperm(len(valid_true_indices))
            train_false_indices = torch.randperm(len(train_false_indices))
            valid_false_indices = torch.randperm(len(valid_false_indices))

            train_indices = torch.randperm([train_true_indices[:self.num_shots]] + [train_false_indices[:self.num_shots]])
            valid_indices = torch.randperm([valid_true_indices[:self.num_shots]] + [valid_false_indices[:self.num_shots]])
            train_data.append(self.train_data[train_indices])
            train_labels.append(self.train_targets[train_indices])
            valid_data.append(self.train_data[valid_indices])
            valid_labels.append(self.train_targets[valid_indices])

        self.order_state = torch.get_rng_state()
        torch.set_rng_state(old_state)

        return {
            'train': (torch.tensor(train_data), torch.tensor(train_labels)),
            'test': (torch.tensor(valid_data), torch.tensor(valid_labels)),
        }

    def batch_data_for_evaluation(self, batch=64):
        train_targets = np.concat((self.train_targets, self.valid_targets))
        temp_train_data = np.concat((self.train_data, self.valid_data))

        train_true_indices = [idx for idx, target in enumerate(train_targets) if target == 1]
        train_false_indices = [idx for idx, target in enumerate(train_targets) if target == 0]

        old_state = torch.get_rng_state()
        torch.set_rng_state(self.choice_state)

        train_data = []
        train_labels = []
        for task_number in range(self.num_tasks):

            train_true_indices = torch.randperm(len(train_true_indices))
            train_false_indices = torch.randperm(len(train_false_indices))

            train_indices = torch.randperm([train_true_indices[:self.num_shots]] + [train_false_indices[:self.num_shots]])
            train_data.append(self.train_data[train_indices])
            train_labels.append(self.train_targets[train_indices])

        self.choice_state = torch.get_rng_state()
        torch.set_rng_state(old_state)

        test_indices = list(range(len(self.test_targets)))
        start_idx = 0
        end_idx = batch
        while start_idx <= len(test_indices):
            test_data = [self.test_text[idx] for idx in test_indices[start_idx:end_idx]]
            test_data = self.prepare_data(test_data)
            test_labels = self.test_targets[test_indices[start_idx:end_idx]]

            yield {
                'train': (torch.tensor(train_data).float(), torch.tensor(train_labels)),
                'test': (torch.tensor(test_data).float(), torch.tensor(test_labels)),
            }
            start_idx = end_idx
            end_idx = end_idx + batch
