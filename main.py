from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import Dataset
from data import ICLDataset, FineTuningDataset, DatasetLoader, PromptDataset, SimilarityICLDataset, InstructionTuningDataset, MetaLearningDataset
from transfer_learning.models import BERTBase, RoBERTaBase, LoRABERTBase, LoRARoBERTaBase
from meta_learning import MAML, FOMAML, Reptile, MetaLearner, ProtoNetMeta, MetaSimpleCnnText, MetaSimpleCnnTextProto, DenseClassifier
import random
import pickle
import argparse
import torch
import numpy as np
import os
import copy
import json
from peft import LoraConfig, PeftModelForCausalLM
from sklearn.metrics import f1_score, accuracy_score
import time
import torch.nn.functional as F

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def parse_results(text, classes):
    pred = -1
    if DATASET in ['cola', 'mrpc'] and PROMPT_FORMAT in [3]:
        for idx, cls in enumerate(classes):
            if (cls.lower() in text.lower()) or (str(idx) in text):
                pred = idx
                break
    else:
        for idx, cls in enumerate(classes):
            if (cls.lower() in text.lower()) or (str(idx) in text):
                if pred == -1:
                    pred = idx
                else:
                    pred = -1
                    break
    return pred


def run_flan_t5(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = ''
        prompt += f'{instructions["instruction"]}\n'
        for sample in context_samples:
            prompt += f'{instructions["sentence_start"]}: {sample[0].strip()}\n{instructions["answer_start"]}: {sample[1].strip()}\n'
    elif DATASET == 'snips' and PROMPT_FORMAT == 3:
        prompt = ''
        for sample in context_samples:
            prompt += f'User: {sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'
    else:
        prompt = ''
        for sample in context_samples:
            prompt += f'{sample[0].strip()}\n{instructions["instruction"]} {sample[1].strip()}\n'
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = []
        for sample in data:
            new_prompt = copy.deepcopy(prompt)
            if PROMPT_FORMAT == 0:
                new_prompt += f'{instructions["sentence_start"]}: {sample.strip()}\n{instructions["answer_start"]}: '
            elif DATASET == 'snips' and PROMPT_FORMAT == 3:
                new_prompt += f'User: {sample.strip()}\n{instructions["instruction"]} '
            else:
                new_prompt += f'{sample.strip()}\n{instructions["instruction"]} '
            final_prompts.append(new_prompt)
        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest', truncation=True).to('cuda')
        out = model.generate(**encoded, max_new_tokens=10)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted


def run_llama2(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        prompt = f'<s>[INST] <<SYS>>\nYou are a helpful assistant that will follow every instruction from the user\n<</SYS>>\n\n{instructions["instruction"]} [/INST] Ok, I will determine the {instructions["task_type"]} of the Sentences you will give me using only the options provided! </s>'
        for sample in context_samples:
            prompt += f'<s>[INST] {instructions["sentence_start"]}: {sample[0].strip()} [/INST] {instructions["answer_start"]}: {sample[1].strip()} </s>'
    else:
        prompt = ''
        for sample in context_samples:
            prompt += f'<s>[INST] {sample[0].strip()}; {instructions["instruction"]} [/INST] {sample[1].strip()} </s>'
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(BATCH_SIZE):
        final_prompts = []
        for sample in data:
            new_prompt = copy.deepcopy(prompt)
            if PROMPT_FORMAT == 0:
                new_prompt += f'<s>[INST] {instructions["sentence_start"]}: {sample.strip()} [/INST] {instructions["answer_start"]}: '
            else:
                prompt += f'<s>[INST] {sample.strip()}; {instructions["instruction"]} '
            final_prompts.append(new_prompt)
        encoded = tokenizer(final_prompts, return_tensors='pt', padding='longest').to('cuda')
        out = model.generate(**encoded, max_new_tokens=10, do_sample=False, num_beams=1, generation_config=generation_config)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('[/INST]')[-1].lower()
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted


def run_mistral(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        messages = [
            {'role': 'user', 'content': instructions['instruction']}, 
            {'role': 'assistant', 'content': f'Ok, I will determine the {instructions["task_type"]} of the Sentences you will give me using only the options provided!'}
        ]
        for sample in context_samples:
            messages.append({'role': 'user', 'content': sample[0]})
            messages.append({'role': 'assistant', 'content': sample[1]})
    else:
        messages = []
        for sample in context_samples:
            messages.append({'role': 'user', 'content': f'{sample[0]} {instructions["instruction"]} '})
            messages.append({'role': 'assistant', 'content': sample[1]})
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            temp_messages = copy.deepcopy(messages)
            if PROMPT_FORMAT == 0:
                temp_messages.append({'role': 'user', 'content': sample})
            else:
                temp_messages.append({'role': 'user', 'content': f'{sample} {instructions["instruction"]} '})
        encoded = tokenizer.apply_chat_template(temp_messages,return_tensors="pt", tokenize=True, add_generation_prompt=True).to('cuda')
        out = model.generate(encoded, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(out)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('[/INST]')[-1]
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted

def run_zephyr(dataset, model, tokenizer):
    instructions = dataset.instructions
    context_samples = dataset.context_samples
    if PROMPT_FORMAT == 0:
        messages = [
            {'role': 'user', 'content': instructions['instruction']}, 
        ]
        for sample in context_samples:
            messages.append({'role': 'user', 'content': sample[0]})
            messages.append({'role': 'assistant', 'content': sample[1]})
    else:
        messages = []
        for sample in context_samples:
            messages.append({'role': 'user', 'content': f'{sample[0]} {instructions["instruction"]} '})
            messages.append({'role': 'assistant', 'content': sample[1]})
    golden = []
    predicted = []
    for data, labels in dataset.batch_data_for_evaluation(1):
        for sample in data:
            temp_messages = copy.deepcopy(messages)
            if PROMPT_FORMAT == 0:
                temp_messages.append({'role': 'user', 'content': sample})
            else:
                temp_messages.append({'role': 'user', 'content': f'{sample} {instructions["instruction"]} '})
        encoded = tokenizer.apply_chat_template(temp_messages,return_tensors="pt", tokenize=True, add_generation_prompt=True).to('cuda')
        out = model.generate(encoded, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(out)

        print(decoded)
        
        predicted_labels = []
        for text in decoded:
            text = text.split('<|assistant|>')[-1]
            pred = parse_results(text, dataset.classes)
            predicted_labels.append(pred)
        print(predicted_labels)
        print(labels)

        predicted.extend(predicted_labels)
        golden.extend(labels)
    return golden, predicted


def run_chatgpt(dataset, investigation_path):
    import openai
    openai.api_key = os.environ['OPEN_AI_KEY']
    openai.organization = os.environ['OPEN_AI_ORGANISATION']

    partial_save_path = os.path.join(investigation_path, 'partial')
    if not os.path.exists(partial_save_path):
        os.makedirs(partial_save_path)

    golden = []
    predicted = []
    idx = 0
    processed = len(os.listdir(partial_save_path))
    for data, labels in dataset.batch_data_for_evaluation(1):
        if idx < processed:
            print(f'Skipping idx: {idx}')
            idx += 1
            continue

        if idx >= len(dataset):
            print('Processed whole dataset')
            break
        prompt = data[0]

        def request_with_checks(prompt):
            success = False
            count = 0
            while not success:
                if count > 0:
                    print(f'Retrying again. Current number of retries: {count}')
                if count >= 5:
                    raise Exception('Too many attempts')
                try:
                    time.sleep(0.5)
                    response = response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that follows all the instructions."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=1,
                    )
                    success = True
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
                    count += 1
            return response

        response = request_with_checks(prompt)
        result = {'predicted': json.loads(str(response)), 'real': labels[0]}
        with open(os.path.join(partial_save_path, f'{idx}.pkl'), 'wb') as file:
            pickle.dump(result, file)
        idx += 1

    for idx, label in enumerate(dataset.targets):

        with open(os.path.join(partial_save_path, f'{idx}.pkl'), 'rb') as file:
            result = pickle.load(file) 

        predicted_label = 1 if 'yes' in result['predicted']['choices'][0]['message']['content'].lower() else 0
        predicted.append(predicted_label)
        golden.append(label)
    
    return golden, predicted



def prompt_icl_experiment(randomness_factor_seeds, model, tokenizer, experiment='icl', investigation_path=None):
    if 'icl' in experiment:
        dataset_constr = SimilarityICLDataset if experiment == 'icl_similarity' else ICLDataset 
        dataset = dataset_constr(
            dataset_name=DATASET,
            train_size=args.train_size,
            num_labelled=args.num_labelled,
            num_labelled_test=args.num_labelled_test,
            split_seed=randomness_factor_seeds['data_split'],
            label_seed=randomness_factor_seeds['label_choice'],
            device=device,
            full_test=FULL_TEST,
            num_shots=args.num_shots,
            num_classes=args.num_classes,
            choice_seed=randomness_factor_seeds['sample_choice'],
            order_seed=randomness_factor_seeds['sample_order'],
            model_name=MODEL,
            prompt_format=PROMPT_FORMAT
        )
    else:
        dataset = PromptDataset(
            dataset_name=DATASET,
            train_size=args.train_size,
            num_labelled=args.num_labelled,
            num_labelled_test=args.num_labelled_test,
            split_seed=randomness_factor_seeds['data_split'],
            label_seed=randomness_factor_seeds['label_choice'],
            device=device,
            full_test=FULL_TEST,
            model_name=MODEL,
            prompt_format=PROMPT_FORMAT
        )

    torch.manual_seed(randomness_factor_seeds['model_randomness'])    
    torch.cuda.manual_seed(randomness_factor_seeds['model_randomness'])
    torch.cuda.manual_seed_all(randomness_factor_seeds['model_randomness'])
    np.random.seed(randomness_factor_seeds['model_randomness'])
    random.seed(randomness_factor_seeds['model_randomness'])

    if MODEL == 'chatgpt':
        return run_chatgpt(dataset, investigation_path)
    else:
        return ICL_MODEL_RUN[f'{MODEL}_{MODEL_SIZE}'](dataset, model, tokenizer)


def instruction_tuning_experiment(randomness_factor_seeds, model_name, tokenizer, investigation_path):
    dataset = InstructionTuningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        split_seed=randomness_factor_seeds['data_split'],
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        model_name=MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()

    tuning_dataset = Dataset.from_dict({'prompt': dataset.prompts, 'label': dataset.targets})

    training_args = TrainingArguments(
        output_dir=investigation_path,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_strategy="no",
        save_strategy="no",
        max_steps=150 if 'steps' in EXPERIMENT_TYPE else -1,
    )

    response_template = "Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tuning_dataset,
        dataset_text_field='prompt',
        data_collator=collator,
    )

    trainer.train()
    model.eval()

    golden = {'prompting': None, 'icl': None}
    predicted = {'prompting': None, 'icl': None}
    for key in ['prompting', 'icl']:
        golden[key], predicted[key] = prompt_icl_experiment(randomness_factor_seeds, model, tokenizer, key)
        score = f1_score(np.array(golden[key]), np.array(predicted[key]), average='macro')
        with open(os.path.join(investigation_path, f'{key}_results.json'), 'w') as file:
            json.dump({'real': golden[key], 'predicted': predicted[key]}, file)

    return golden, predicted


def ft_experiment(randomness_factor_seeds):
    if 'lora' in MODEL:
        tokenizer = AutoTokenizer.from_pretrained(f'{MODEL.split("_")[1] if "lora" in MODEL else MODEL}-{MODEL_SIZE}{"-uncased" if MODEL.split("_")[1] == "bert" else ""}', return_dict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f'{MODEL}-{MODEL_SIZE}{"-uncased" if MODEL == "bert" else ""}', return_dict=False)
    dataset = FineTuningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        split_seed=randomness_factor_seeds['data_split'],
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    loader = DatasetLoader('sst2', BATCH_SIZE, dataset, randomness_factor_seeds['sample_order'])
    trainloader = loader.trainloader()
    testloader = loader.testloader()

    net = FT_MODELS[MODEL](dataset.n_classes, randomness_factor_seeds['model_initialisation'], randomness_factor_seeds['model_randomness'], True)
    net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.train()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, data in enumerate(trainloader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = net(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
    
    golden = []
    predictions = []
    
    net.eval()
    for batch_idx, data in enumerate(testloader):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        outputs = net(ids, mask, token_type_ids)

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())
        golden.extend(data['targets'].tolist())
    return golden, predictions


def meta_learning_experiment(randomness_factor_seeds):
    META_MODEL_MAPPING = {
        'maml': MAML,
        'fomaml': FOMAML,
        'protonet': ProtoNetMeta,
        'reptile': Reptile,
    }

    BASE_MODEL_MAPPING = {
        'protonet': MetaSimpleCnnTextProto,
        'maml': MetaSimpleCnnText,
        'fomaml': MetaSimpleCnnText,
        'reptile': MetaSimpleCnnText,
    }

    OPTIMIZER_MAPPING = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }


    dataset = MetaLearningDataset(
        dataset_name=DATASET,
        train_size=args.train_size,
        num_labelled=args.num_labelled,
        num_labelled_test=args.num_labelled_test,
        split_seed=randomness_factor_seeds['data_split'],
        label_seed=randomness_factor_seeds['label_choice'],
        device=device,
        full_test=FULL_TEST,
        max_len=MAX_LEN,
        num_tasks=args.num_tasks,
        num_shots=args.num_shots, 
        choice_seed=randomness_factor_seeds['sample_choice'],
        order_seed=randomness_factor_seeds['sample_order'],
    )
    base_model = BASE_MODEL_MAPPING[args.model](
        sentence_length=MAX_LEN,
        embedding_dim=768,
        n_filters=128,
        filter_size=5,
        pool_size=2,
        hidden_size=128,
        num_classes=2,
        init_seed=randomness_factor_seeds['model_initialisation'],
        randomness_seed=randomness_factor_seeds['model_randomness']
    )

    meta_model = META_MODEL_MAPPING[args.model](
        model=base_model,
        params={
            'meta_lr': args.meta_lr,
            'base_lr': args.lr,
            'meta_optimizer': OPTIMIZER_MAPPING[args.meta_optimizer],
            'base_optimizer': OPTIMIZER_MAPPING[args.base_optimizer],
            'loss_function': F.cross_entropy,
            'inner_iterations': args.inner_iterations,
            'lr_scheduler': torch.optim.lr_scheduler.StepLR
        },
        devide=device
    )

    meta_learner = MetaLearner(meta_model)

    meta_learner.train(
        dataset,
        num_batches=args.num_batches,
        epochs=args.num_epochs,
        verbose=False
    )

    dataset.num_shots = args.num_shots_test
    predicted, golden = meta_learner.evaluate_in_batch(dataset, 128)
    return predicted, golden



parser = argparse.ArgumentParser()
# Meta
parser.add_argument('--experiment_name', default='investigation_experiments', type=str, help='Directory to save experiments to')
parser.add_argument('--configuration_name', default='stability', type=str, help='Further distinction for the save directory')
parser.add_argument('--experiment_type', default='icl', type=str, choices=['finetuning', 'prompting', 'icl', 'icl_similarity', 'instruction_tuning' 'instruction_tuning_steps', 'meta_learning'], help='Type of experiment to run')
parser.add_argument('--full_test', default=1, type=int, help='Whether to use whole test dataset (Yes (default): 1; No: 0). If "No" and "num_labelled_test" is not set then uses same number of labelled samples as defined by num_labelled')
parser.add_argument('--regenerate', default=0, type=int, help='Whether to calculate every result again or continue from checkpoint (Yes: 1; No (default): 0).')
# General training args
parser.add_argument('--factor', default='golden_model', type=str, choices=['golden_model', 'data_split', 'label_choice', 'sample_choice', 'sample_order', 'model_initialisation', 'model_randomness'], help='Randomness factor to investigate.')
parser.add_argument('--num_shots', default=4, type=int, help='Number of samples to use as in-context examples in in-context learning or in different tasks in meta-learning.')
parser.add_argument('--dataset', default='sst2', type=str, choices=['sst2', 'mrpc', 'cola', 'rte', 'boolq', 'trec', 'ag_news', 'db_pedia', 'snips'], help='Dataset to use for investigation.')
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes in dataset.')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_size', default=0.8, type=float)
parser.add_argument('--num_labelled', default=1000, type=int)
parser.add_argument('--num_labelled_test', default=1000, type=int)
parser.add_argument('--model', default='flan-t5', type=str, choices=['bert', 'roberta', 'flan-t5', 'llama2', 'chatgpt', 'protonet', 'maml', 'fomaml', 'reptile', 'mistral', 'zephyr', 'lora_bert', 'lora_roberta'])
parser.add_argument('--model_size', default='base', type=str, choices=['base'])
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=5, type=int, help='Total number of epochs to train for')
parser.add_argument('--max_len', default=20, type=int, help='Maximal length of input for fine-tuning experiments')
parser.add_argument('--prompt_format', default=0, type=int, help='Which prompt format to use')
# Seeds
parser.add_argument('--mitigation_seed', default=42, type=int, help='Seed for generating seeds for investigation')
parser.add_argument('--investigation_seed', default=27, type=int, help='Seed for generating seeds for investigation')
# Investigation
parser.add_argument('--investigation_runs', default=10, type=int, help='Number of different configurations for investigating chosen randomness factor.')
parser.add_argument('--mitigation_runs', default=100, type=int, help='Number of different configurations for mitigating other randomness factors.')
# Meta-Learning specific
parser.add_argument('--num_tasks', default=16, type=int, help='Number of different tasks to use in meta-learning experiments.')
parser.add_argument('--meta_optimizer', default='Adam', type=str, choices=['Adam', 'SGD'], help='Optimizer to use for the meta model in meta-learning.')
parser.add_argument('--base_optimizer', default='SGD', type=str, choices=['Adam', 'SGD'], help='Optimizer to use for the base model in meta-learning.')
parser.add_argument('--num_shots_test', default=15, type=int, help='Number of samples in evaluation tasks in meta-learning.')
parser.add_argument('--num_batches', default=64, type=int, help='Number of batches to run in meta-learning.')
parser.add_argument('--meta_lr', default=1e-5, type=float)
parser.add_argument('--inner_iterations', default=1, type=int, help='Number of inner adaptation steps')

parser.add_argument('-f')
args = parser.parse_args()

device = torch.device('cuda')
FT_MODELS = {
    'bert':  BERTBase,
    'roberta':  RoBERTaBase,
    'lora_bert': LoRABERTBase,
    'lora_roberta': LoRARoBERTaBase,
}

ICL_MODELS = {
    'flan-t5_base': 'google/flan-t5-base',
    'llama2_base': 'meta-llama/Llama-2-13b-chat-hf',
    'mistral_base': 'mistralai/Mistral-7B-Instruct-v0.1',
    'zephyr_base': 'HuggingFaceH4/zephyr-7b-alpha'
}

ICL_MODEL_RUN = {
    'flan-t5_base': run_flan_t5,
    'llama2_base': run_llama2,
    'chatgpt_base': run_chatgpt,
    'mistral_base': run_mistral,
    'zephyr_base': run_zephyr,
}

EXPERIMENT_TYPE = args.experiment_type
FULL_TEST = args.full_test == 1
MAX_LEN = args.max_len
PROMPT_FORMAT = args.prompt_format

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = not torch.backends.cudnn.deterministic

MODEL = args.model
MODEL_SIZE = args.model_size
FACTOR = args.factor
DATASET = args.dataset
RESULTS_PATH = os.path.join('results', f'{args.experiment_name}', f'{EXPERIMENT_TYPE}_{MODEL}_{MODEL_SIZE}', args.configuration_name, DATASET, FACTOR)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

BATCH_SIZE = args.batch_size # 64
NUM_EPOCHS = args.num_epochs # 5
LEARNING_RATE = args.lr # 1e-5

if os.path.exists(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl')):
    with open(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl'), 'rb') as file:
        print(f'Loading mitigation seeds:')
        mitigation_seeds = pickle.load(file)
        print(mitigation_seeds)
        print(f'Length of mitigation seeds: {len(mitigation_seeds)}')
if not os.path.exists(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl')) or len(mitigation_seeds) != args.mitigation_runs:
    print(f'Generating new mitigation seeds of length: {args.mitigation_runs}')
    random.seed(args.mitigation_seed)
    mitigation_seeds = [random.randint(1, 100000) for _ in range(args.mitigation_runs)]
    print(mitigation_seeds)
    with open(os.path.join(RESULTS_PATH, 'mitigation_seeds.pkl'), 'wb') as file:
        pickle.dump(mitigation_seeds, file)


if os.path.exists(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl')):
    with open(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl'), 'rb') as file:
        print(f'Loading investigation seeds:')
        investigation_seeds = pickle.load(file)
        print(investigation_seeds)
        print(f'Length of investigation seeds: {len(investigation_seeds)}')
if not os.path.exists(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl')) or len(investigation_seeds) != args.investigation_runs:
    print(f'Generating new investigation seeds of length: {args.investigation_runs}')
    random.seed(args.investigation_seed)
    investigation_seeds = [random.randint(1, 100000) for _ in range(args.investigation_runs)]   
    print(investigation_seeds)
    with open(os.path.join(RESULTS_PATH, 'investigation_seeds.pkl'), 'wb') as file:
        pickle.dump(investigation_seeds, file)


if MODEL == 'chatgpt':
    model_name = MODEL

elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'

elif EXPERIMENT_TYPE in ('icl', 'prompting', 'icl_similarity'):
    model_name = ICL_MODELS[f'{MODEL}_{MODEL_SIZE}']

    if MODEL == 'llama2':
        access_token = os.environ['HUGGINGFACE_TOKEN']
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=access_token)
        
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, token=access_token)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        generation_config = model.generation_config
        generation_config.num_beams = 1
        generation_config.max_new_tokens = 4
        generation_config.do_sample = False
        generation_config.temperature = None
        model.eval()
    elif MODEL in ['mistral', 'zephyr']:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = 'left'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()
    model.eval()

else:
    if 'lora' in MODEL:
        model_name = f'{MODEL.split("_")[1]}-{MODEL_SIZE}{"-uncased" if MODEL.split("_")[1] == "bert" else ""}'
    else:
        model_name = f'{MODEL}-{MODEL_SIZE}{"-uncased" if MODEL == "bert" else ""}'

randomness_factors = ['data_split', 'label_choice', 'sample_choice', 'sample_order', 'model_initialisation', 'model_randomness']

print(f'Running investigation for factor {FACTOR}')

for mit_idx, mitigation_seed in enumerate(mitigation_seeds):
    print(f'Running mitigation number {mit_idx} with seed {mitigation_seed}')
    mitigation_path = os.path.join(RESULTS_PATH, f'mitigation_{mit_idx}')
    if not os.path.exists(mitigation_path):
        os.makedirs(mitigation_path)
    
    randomness_factor_seeds = {factor: mitigation_seed for factor in randomness_factors}

    for inv_idx, investigation_seed in enumerate(investigation_seeds):
        print(f'Running investigation number {inv_idx} with seed {investigation_seed}')
        investigation_path = os.path.join(mitigation_path, f'investigation_{inv_idx}')
        if os.path.exists(os.path.join(investigation_path, 'results.json')) and args.regenerate == 0:
            print(f'Investigation number {inv_idx} already exists under mitigation {mit_idx}. Skipping!')
            continue
        if not os.path.exists(investigation_path):
            os.makedirs(investigation_path)
        
        if FACTOR != 'golden_model':
            randomness_factor_seeds[FACTOR] = investigation_seed

        if EXPERIMENT_TYPE in ['finetuning']:
            print(f'Running fine-tuning experiments!')
            golden, predicted = ft_experiment(randomness_factor_seeds)
        elif EXPERIMENT_TYPE == 'meta_learning':
            golden, predicted = meta_learning_experiment(randomness_factor_seeds)
        elif EXPERIMENT_TYPE in ['instruction_tuning', 'instruction_tuning_steps']:
            golden, predicted = instruction_tuning_experiment(randomness_factor_seeds, model_name, tokenizer, investigation_path)
        elif MODEL == 'chatgpt':
            golden, predicted = prompt_icl_experiment(randomness_factor_seeds, None, None, EXPERIMENT_TYPE, investigation_path)
        else:
            golden, predicted = prompt_icl_experiment(randomness_factor_seeds, model, tokenizer, EXPERIMENT_TYPE)
        
        print(np.mean(np.array(golden) == np.array(predicted)))
        results = copy.deepcopy(randomness_factor_seeds)
        results['real'] = golden
        results['predicted'] = predicted
        results['base_model'] = model_name
        results['mitigation_idx'] = mit_idx
        results['investigation_idx'] = inv_idx

        with open(os.path.join(investigation_path, 'results.json'), 'w') as file:
            json.dump(results, file)

