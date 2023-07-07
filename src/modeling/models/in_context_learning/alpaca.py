import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
import numpy as np


class Alpaca():
    def __init__(self, device=None):
        model_name = 'decapoda-research/llama-7b-hf'
        weights_name = 'tloen/alpaca-lora-7b'

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map='auto')
        self.model = PeftModel.from_pretrained(self.model, weights_name, torch_dtype=torch.float16)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.model.eval()

        self.generation_config = GenerationConfig(
            temperature=0,
            top_p=0.75,
            top_k=10,
            num_beams=1
        )

        self.device = device

    def evaluate_in_batch(self, dataloader, batch_size):
        self.model.eval()
        predicted = []
        labels = []

        for batch_idx, batch in enumerate(dataloader.batch_data_for_evaluation(batch_size)):
            inputs, targets = batch

            max_size = -1
            processed_inputs = []
            for input in inputs:
                input = self.tokenizer(input)
                input_ids = input['input_ids']
                max_size = max(max_size, len(input_ids))
                processed_inputs.append(input_ids)

            # Pad from left the input
            input_ids = []
            for input in processed_inputs:
                if len(input) < max_size:
                    input = np.vstack((
                        np.zeros((max_size - len(input))),
                        input
                    ))
                input_ids.append(input)

            input_ids = torch.tensor(np.array(input_ids)).to(self.device)

            with torch.no_grad():
                generation_output = self.model.generate(input_ids=input_ids, generation_config=self.generation_config, return_dict_in_generate=True, max_new_tokens=25)
                s = generation_output.sequences
                for idx, sequence in enumerate(s):
                    output = self.tokenizer.decode(sequence)
                    print(f'{output}')
                    final_output = output.split('### Response:')[1].strip()
                    if 'Yes' in final_output:
                        predicted.append(1)
                    else:
                        predicted.append(0)
                    labels.append(targets[idx])

    def evaluate(self, dataloader):
        self.model.eval()
        predicted = []
        labels = []

        for input, target in dataloader.iter_data_for_evaluation():
            input = self.tokenizer(input, return_tensor='pt')
            input_ids = input['input_ids'].to(self.device)

            with torch.no_grad():
                generated_output = self.model.generate(input_ids=input_ids, generation_config=self.generation_config, return_dict_in_generate=True, max_new_tokens=10)
                s = generated_output.sequences[0]
                output = self.tokenizer.decode(s)
                print(f'{output}')
                final_output = output.split('### Response:')[1].strip()
                if 'Yes' in final_output:
                    predicted.append(1)
                else:
                    predicted.append(0)
                labels.append(target)


        