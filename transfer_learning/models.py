import torch
import random
import numpy as np
from transformers import BertModel, RobertaModel, BertForSequenceClassification, BertConfig, RobertaConfig, RobertaForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

class DeterministicModel():
    def __init__(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

    def set_rng_state(self, seed):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state

    def restore_rng_state(self, states):
        old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state = states

        torch.set_rng_state(old_torch_state)
        torch.cuda.set_rng_state(old_torch_cuda_state)
        np.random.set_state(old_numpy_state)
        random.setstate(old_random_state)

    def get_rng_state(self):
        old_torch_state = torch.get_rng_state()
        old_torch_cuda_state = torch.cuda.get_rng_state()
        old_numpy_state = np.random.get_state()
        old_random_state = random.getstate()
        return old_torch_state, old_torch_cuda_state, old_numpy_state, old_random_state



class BERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(BERTBase, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)


        _, bert_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids
        )
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class LoRABERTBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(LoRABERTBase, self).__init__()
        bert_config = BertConfig(name_or_path='bert-base-uncased', num_labels=n_classes)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        self.bert = BertForSequenceClassification(bert_config)
        self.bert = get_peft_model(self.bert, lora_config)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids
        )

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output['logits']

class RoBERTaBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(RoBERTaBase, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base', return_dict=False)
        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)


        _, bert_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids
        )
        output = self.dropout(bert_output)
        output = self.output(output)

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output

class LoRARoBERTaBase(torch.nn.Module, DeterministicModel):

    def __init__(self, n_classes, init_seed=0, dropout_seed=0, trainable=True):
        self.name = 'bert-base'
        states = self.set_rng_state(init_seed)
        super(LoRARoBERTaBase, self).__init__()
        bert_config = RobertaConfig(name_or_path='roberta-base', num_labels=n_classes)
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        self.bert = RobertaForSequenceClassification(bert_config)
        self.bert = get_peft_model(self.bert, lora_config)
        self.restore_rng_state(states)

        states = self.set_rng_state(dropout_seed)
        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

    def forward(self, input_ids, attention_mask, token_type_ids):
        states = self.get_rng_state()
        self.restore_rng_state(self.dropout_states)

        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          token_type_ids=token_type_ids
        )

        self.dropout_states = self.get_rng_state()
        self.restore_rng_state(states)

        return output['logits']