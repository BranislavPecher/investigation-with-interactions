import torch
import copy
import os
import pickle
import math
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, prompt_format=0):
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
        self.prompt_format = prompt_format

        self.text, self.targets = self.initialise_dataset_from_huggingface()
        self.num_classes = len(self.classes)
        self.split_train_test()
        self.select_labelled_data()


    def initialise_dataset_from_huggingface(self):
        if self.dataset_name == 'sst2':
            print('Using SST-2 dataset.')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            if self.prompt_format in [0, 1, 2]:
                self.classes = ['negative', 'positive']
            elif self.prompt_format in [3]:
                self.classes = ['terrible', 'great']
            else:
                raise NotImplemented
            # self.classes = ['No', 'Yes']
            return data.sentence.tolist(), data.label.tolist()
        elif self.dataset_name == 'cola':
            print(f'Using cola dataset')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            if self.prompt_format in [0, 1]:
                self.classes = ['No', 'Yes']
            elif self.prompt_format in [2]:
                self.classes = ['Yes', 'No']
            elif self.prompt_format in [3]:
                self.classes = ['not acceptable', 'acceptable']
            else:
                raise NotImplemented
            return data.sentence.tolist(), data.label.tolist()
        elif self.dataset_name == 'mrpc':
            print('Using mrpc')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation']), pd.DataFrame(dataset['test'])])
            texts = [f'Sentence 1: {sent1}; Sentence 2: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            if self.prompt_format in [0, 1]:
                self.classes = ['No', 'Yes']
            elif self.prompt_format in [2]:
                self.classes = ['Yes', 'No']
            elif self.prompt_format in [3]:
                self.classes = ['not equivalent', 'equivalent']
            else:
                raise NotImplemented
            return texts, data.label.tolist()
        elif self.dataset_name == 'rte':
            print('Using rte')
            dataset = load_dataset('glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Premise: {sent1}; Hypothesis: {sent2}' for sent1, sent2 in zip(data.sentence1.tolist(), data.sentence2.tolist())]
            self.classes = ['not entailment', 'entailment']
            return texts, data.label.tolist()
        elif self.dataset_name == 'boolq':
            print('Using BoolQ dataset')
            dataset = load_dataset('super_glue', self.dataset_name)
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['validation'])])
            texts = [f'Question: {question}\nPassage: {passage}' for question, passage in zip(data.question.tolist(), data.passage.tolist())]
            self.classes = ['No', 'Yes']
            return texts, data.label.tolist()
        elif self.dataset_name == 'trec':
            print('Using TREC dataset')
            dataset = load_dataset('trec')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            self.classes = ['Expression', 'Entity', 'Description', 'Human', 'Location', 'Number']
            return data.text.tolist(), data.coarse_label.tolist()
        elif self.dataset_name == 'ag_news':
            print('Using AG News dataset')
            dataset = load_dataset('ag_news')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            self.classes = ['World', 'Sports', 'Business', 'Science and Technology']
            return data.text.tolist(), data.label.tolist()
        elif self.dataset_name == 'snips':
            print('Using SNIPS dataset')
            dataset = load_dataset('benayas/snips')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            mapper = {
                'AddToPlaylist':            0,
                'GetWeather':               1,
                'SearchScreeningEvent':     2,
                'PlayMusic':                3,
                'SearchCreativeWork':       4,
                'RateBook':                 5,
                'BookRestaurant':           6,
            }
            data['label'] = data['category'].apply(lambda x: mapper[x])
            self.classes = ['Playlist', 'Weather', 'Event', 'Musing', 'Creative Work', 'Rate Book', 'Book Restaurant']
            return data.text.tolist(), data.label.tolist()
        elif self.dataset_name == 'db_pedia':
            print('Using DB Pedia dataset')
            dataset = load_dataset('fancyzhx/dbpedia_14')
            data = pd.concat([pd.DataFrame(dataset['train']), pd.DataFrame(dataset['test'])])
            self.classes = ['Company', 'Educational Institution', 'Artist', 'Athlete', 'Office Holder', 'Transportation', 'Building', 'Natural Place', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Written Work']
            return data.content.tolist(), data.label.tolist()
        else:
            raise NotImplemented('The dataset cannot be initiated!')


    def split_train_test(self, train_test_indices=None):
        if train_test_indices is None:
            old_state = torch.get_rng_state()
            torch.manual_seed(self.split_seed)
            indices = list(range(len(self.text)))
            self.train_indices, self.test_indices = train_test_split(indices, train_size=self.train_size, random_state=self.split_seed, stratify=self.targets)
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

            to_select = math.ceil(self.num_labelled / self.num_classes)

            targets = np.array(self.train_targets)

            texts = []
            labels = []
            train_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                train_indices.extend(inds)
                for idx in inds:
                    texts.append(self.train_text[idx])
                    labels.append(self.train_targets[idx])
            indices = torch.randperm(len(labels))
            self.train_text = [texts[idx] for idx in indices]
            self.train_targets = [labels[idx] for idx in indices]
            self.train_indices = train_indices

            print(f'Number of selected Train samples: {len(self.train_targets)}')

            torch.set_rng_state(old_state)
        
        if not self.full_test and self.num_labelled_test > 0:

            old_state = torch.get_rng_state()
            torch.manual_seed(self.label_seed)

            to_select = math.ceil(self.num_labelled_test / self.num_classes)

            targets = np.array(self.test_targets)

            texts = []
            labels = []
            test_indices = []
            for cls in range(self.num_classes):
                inds = np.argwhere(targets == cls).reshape(-1)
                indices = torch.randperm(len(inds))
                inds = inds[indices]
                inds = inds[:to_select]
                test_indices.extend(inds)
                for idx in inds:
                    texts.append(self.test_text[idx])
                    labels.append(self.test_targets[idx])
            indices = torch.randperm(len(labels))
            self.test_text = [texts[idx] for idx in indices]
            self.test_targets = [labels[idx] for idx in indices]
            self.test_indices = test_indices
            print(len(self.test_text))

            print(f'Number of selected Test samples: {len(self.test_targets)}')

            torch.set_rng_state(old_state)

    def __prepare_dataset_keywords(self):
        options = ''
        for idx, text in enumerate(self.classes):
            options += f' {idx + 1}) {text}'

        prompt = self.prompt_format

        if self.dataset_name == 'sst2':
            if prompt == 0:
                instruction = f'Determine sentiment of the sentence using following options:{options}'
            elif prompt == 1:
                instruction = 'Sentiment?'
            elif prompt == 2:
                instruction = 'Senstiment is'
            elif prompt == 3:
                instruction = 'It was'
            else:
                raise NotImplemented
            sentence_start = 'Sentence'
            answer_start = 'Answer'
            task_type = 'sentiment'
        elif self.dataset_name == 'cola':
            if prompt == 0:
                instruction = f'Determine grammatical acceptability of the Sentence using following options:{options}'
            elif prompt == 1:
                instruction = 'Grammatically acceptable?'
            elif prompt == 2:
                instruction = 'Grammar problems?'
            elif prompt == 3:
                instruction = 'It is'
            else:
                raise NotImplemented
            sentence_start = 'Sentence'
            answer_start = 'Answer'
            task_type = 'grammatical acceptability'
        elif self.dataset_name == 'mrpc':
            if prompt == 0:
                instruction = f'Determine whether the Sentence Pair is semantically equivalent using following options:{options}'
            elif prompt == 1:
                instruction = 'Semantically equivalent sentences?'
            elif prompt == 2:
                instruction = 'Semantically different sentences?'
            elif prompt == 3:
                instruction = 'Sentences are'
            else:
                raise NotImplemented
            sentence_start = 'Sentence Pair'
            answer_start = 'Answer'
            task_type = 'semantical equivalence'
        elif self.dataset_name == 'rte':
            instruction = f'Determine whether the Premise entails the Hypothesis using following options:{options}'
            sentence_start = ''
            answer_start = 'Answer'
            task_type = 'entailment'
        elif self.dataset_name == 'boolq':
            instruction = f'Determine whether the Passage contains Answer to the Question using following options:{options}'
            sentence_start = ''
            answer_start = 'Answer'
            task_type = 'presence'
        elif self.dataset_name in ['trec', 'ag_news', 'db_pedia']:
            if prompt == 0:
                instruction = f'Determine topic of the sentence using following options:{options}'
            elif prompt == 1:
                instruction = 'Topic?'
            elif prompt == 2:
                instruction = 'Topic is'
            elif prompt == 3:
                instruction = 'This is about'
            else:
                raise NotImplemented
            sentence_start = 'Sentence'
            answer_start = 'Answer'
            task_type = 'topic'
        elif self.dataset_name == 'snips':
            if prompt == 0:
                instruction = f'Determine intent of the sentence using following options:{options}'
            elif prompt == 1:
                instruction = 'Intent?'
            elif prompt == 2:
                instruction = 'Intent is'
            elif prompt == 3:
                instruction = 'User requested'
            else:
                raise NotImplemented
            sentence_start = 'Sentence'
            answer_start = 'Answer'
            task_type = 'intent'


        return instruction, sentence_start, answer_start, task_type


class ICLDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, num_shots=2, num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5', prompt_format=0):
        super(ICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test, prompt_format)

        self.num_shots = num_shots
        
        self.choice_seed = choice_seed
        self.order_seed = order_seed

        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()


    def prepare_dataset_for_use(self):
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        instructions, context_samples = self.__prepare_prompt(texts, targets)
        return instructions, context_samples

    
    def batch_data_for_evaluation(self, batch=64):
        start_idx = 0
        end_idx = batch

        while start_idx < len(self.test_text):
            data = self.test_text[start_idx : end_idx]
            labels = self.test_targets[start_idx : end_idx]

            yield data, labels

            start_idx = end_idx
            end_idx += batch


    def __len__(self):
        return len(self.test_text)
    

    def __choose_shots(self):
        to_choose = int(self.num_shots)
        
        old_state = torch.get_rng_state()
        torch.manual_seed(self.choice_seed)

        targets = np.array(self.train_targets)

        texts = []
        labels = []
        for cls in range(self.num_classes):
            inds = np.argwhere(targets == cls).reshape(-1)
            indices = torch.randperm(len(inds))
            inds = inds[indices]
            inds = inds[:to_choose]
            for idx in inds:
                texts.append(self.train_text[idx])
                labels.append(self.train_targets[idx])

        torch.set_rng_state(old_state)

        return texts, labels

    def __sample_reorder(self, texts, targets):
        old_state = torch.get_rng_state()
        torch.manual_seed(self.order_seed)

        indices = torch.randperm(len(texts))
        texts = [texts[idx] for idx in indices]
        targets = [targets[idx] for idx in indices]

        torch.set_rng_state(old_state)

        return texts, targets

    def __prepare_prompt(self, texts, targets):
        instruction, sentence_start, answer_start, task_type = self.__prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = [(texts[idx], self.classes[targets[idx]]) for idx in range(len(targets))]
        return instructions, context_samples

    


class SimilarityICLDataset(ICLDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, num_shots=4, num_classes=2, choice_seed=0, order_seed=0, model_name='flan-t5', prompt_format=0):
        super(SimilarityICLDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test, num_shots, num_classes, choice_seed, order_seed, model_name, prompt_format)

        self.num_shots = num_shots
        
        self.choice_seed = choice_seed
        self.order_seed = order_seed

        with open(os.path.join('data', f'{dataset_name}_embeddings.pkl'), 'rb') as file:
            self.embeddings = pickle.load(file)

        self.model_name = model_name
        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        texts, targets = self.__choose_shots()
        texts, targets = self.__sample_reorder(texts, targets)
        prompts, targets = self.__prepare_prompt(texts, targets)
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


class PromptDataset(TextDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, model_name='flan-t5', prompt_format=0):
        super(PromptDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test, prompt_format)
        self.model_name = model_name

        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instructions, context_samples = self.__prepare_prompt()
        return instructions, context_samples

    
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
        return len(self.test_text)

    
    def __prepare_prompt(self):
        instruction, sentence_start, answer_start, task_type = self.__prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = []
        return instructions, context_samples


class InstructionTuningDataset(ICLDataset):
    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, model_name='flan-t5', prompt_format=0):
        super(InstructionTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test, 0, 0, 0, 0, model_name, prompt_format)
        self.model_name = model_name

        self.instructions, self.context_samples = self.prepare_dataset_for_use()

    def prepare_dataset_for_use(self):
        instructions, context_samples = self.__prepare_prompt(self.train_text, self.train_targets)
        return instructions, context_samples

    
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
        instruction, sentence_start, answer_start, task_type = self.__prepare_dataset_keywords()
        instructions = {
            'instruction': instruction,
            'sentence_start': sentence_start,
            'answer_start': answer_start,
            'task_type': task_type
        }

        context_samples = [(texts[idx], self.classes[targets[idx]]) for idx in range(len(targets))]
        return instructions, context_samples


class FineTuningDataset(TextDataset):

    def __init__(self, dataset_name, train_size=0.8, num_labelled=1000, num_labelled_test=1000, split_seed=0, label_seed=0, device=None, full_test=True, tokenizer=None, max_len=50):
        super(FineTuningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)
        self.tokenizer = tokenizer
        self.train = True
        self.max_len = max_len

        self.n_classes = self.num_classes

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
        super(MetaLearningDataset, self).__init__(dataset_name, train_size, num_labelled, num_labelled_test, split_seed, label_seed, device, full_test)

        self.num_tasks = num_tasks
        self.num_shots = num_shots

        self.train = True
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
