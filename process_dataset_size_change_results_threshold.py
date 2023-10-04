import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
from data import TextDataset

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_results(sizes, l3d_type, model_name, runs=100, special_type=None, to_load=None):
    means = []
    stds = []
    for size in sizes:
        golden_model = []
        all = failed = 0
        for mitigation in range(runs):
            if special_type is None:
                path = os.path.join('results', f'dataset_size_change_{l3d_type}_{model_name}_base', f'num_samples_{size}', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
            else:
                path = os.path.join('results', f'dataset_size_change_{l3d_type}_{model_name}_base', f'num_samples_{special_type}', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                if to_load is None:
                    score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
                else:
                    score = f1_score(np.array(data['real'][to_load]), np.array(data['predicted'][to_load]), average='macro')
                if score < .5:
                    failed += 1
                all += 1
                golden_model.append(score * 100)
            except Exception as e:
                print(e)
                continue

        print(f'Golden model (size={size}):')
        print(f'Failed percentage: {failed / all * 100}')
        print(f'Mean: {np.mean(golden_model)}')
        print(f'Std: {np.std(golden_model)}')
        print(f'Min: {np.min(golden_model)}')
        print(f'Max: {np.max(golden_model)}')
        print()
        means.append(np.mean(golden_model))
        stds.append(np.std(golden_model))
    means = np.array(means)
    stds = np.array(stds)
    return means, stds


def sst2_flant5_prompting_results(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'sst2'
    sizes = [50, 100, 150, 200, 250, 300]

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    color_idx = 0
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)
        color_idx += 1

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')
        color_idx += 1

    ax.legend()
    ax.set_title('Finetuning vs. Flan-T5')


def sst2_llama_chatgpt_prompting_results(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'sst2'
    sizes = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600]

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['llama2']:
        for l3d_type, display_name in [('prompting', 'Prompting')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('prompting', 'Prompting')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('Finetuning vs. LLaMA-2/ChatGPT Prompting')


def sst2_llama_chatgpt_icl_results(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'sst2'
    sizes = [2000, 2500, 3000, 4000, 5000, 6000]

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['llama2']:
        for l3d_type, display_name in [('icl', 'ICL')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 5000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('icl', 'ICL')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('Finetuning vs. LLaMA-2/ChatGPT ICL')



def sst2_instruction_tuning(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'sst2'
    sizes = [8000, 10000, 12000, 15000, 20000]

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                color = mapping['flan-t5-instruction']
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('Finetuning vs. Instruction Tuning')


def mrpc_llama(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'mrpc'
    sizes = [50, 100, 150, 200, 250, 300, 350, 500]

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['llama2']:
        for l3d_type, display_name in [('prompting', 'Prompting')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    for model_name in ['llama2']:
        for l3d_type, display_name in [('icl', 'ICL')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 250)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('Finetuning vs. LLaMA-2 Prompting/ICL')


def mrpc_roberta(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'mrpc'
    sizes = [200, 250, 300, 350, 400, 500]

    configurations = [('roberta', 'finetuning', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                color = mapping['flan-t5-instruction']
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('RoBERTa vs. Flan-T5/ChatGPT Prompting/ICL vs. Instruction Tuning')


def mrpc_bert(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'mrpc'
    sizes = [500, 600, 700, 800, 900, 1000, 1100, 1200]

    configurations = [('bert', 'finetuning', 'BERT FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                color = mapping['flan-t5-instruction']
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('BERT vs. Flan-T5/ChatGpt Prompting/ICL vs. Instruction Tuning')


def boolq_flanicl(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'boolq'
    sizes = [1000, 2000, 2500, 3000]

    configurations = [('roberta', 'finetuning_2', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('icl', 'ICL')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                color = mapping['flan-t5-instruction']
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('icl', 'ICL')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['llama2']:
        for l3d_type, display_name in [('prompting', 'Prompting')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('RoBERTa vs. LLaMA-2 Prompting vs. Flan-T5/Instruction Tuning ICL')

def boolq_rest(mapping, ax):
    global DATASET_PATH
    DATASET_PATH = 'boolq'
    sizes = [3000, 4000, 5000, 6000, 7000]

    configurations = [('roberta', 'finetuning_2', 'RoBERTa FineTuning')]
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        color = mapping[model_name]
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=color)
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color)

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('prompting', 'Prompting')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                color = mapping['flan-t5-instruction']
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('prompting', 'Prompting')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=color, linestyle='solid')

    for model_name in ['llama2']:
        for l3d_type, display_name in [('icl', 'ICL')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, 5000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')
    
    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            color = mapping[model_name]
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=color, linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=color, linestyle='solid')

    ax.legend()
    ax.set_title('RoBERTa vs. Prompting/ICL vs. Instruction Tuning')


models = ['bert', 'roberta', 'flan-t5-instruction', 'flan-t5', 'llama2', 'chatgpt', 'random']
clrs = sns.color_palette("deep", len(models))
mapping = {models[idx]: clrs[idx] for idx in range(len(models))}

plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.95, top=0.92, hspace=0.125, wspace=0.15)
fig.subplots_adjust(left=0.06, bottom=0.06, right=0.95, top=0.92, hspace=0.125, wspace=0.15)
fig.suptitle('Finetuning vs. Prompting/ICL vs. Instruction Tuning - SST2', size=14)
fig.supxlabel('Train Dataset Size', size=14)
fig.supylabel('F1 Macro (+- STD)', size=14)

ax = fig.add_subplot(2, 2, 1)
sst2_flant5_prompting_results(mapping, ax)
ax = fig.add_subplot(2, 2, 2)
sst2_llama_chatgpt_prompting_results(mapping, ax)
ax = fig.add_subplot(2, 2, 3)
sst2_llama_chatgpt_icl_results(mapping, ax)
ax = fig.add_subplot(2, 2, 4)
sst2_instruction_tuning(mapping, ax)
plt.savefig(os.path.join('visualisations', 'sst2', f'sst2_threshold_image_multi.png'), dpi=300, bbox_inches='tight')

plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.95, top=0.92, hspace=0.125, wspace=0.15)
fig.subplots_adjust(left=0.06, bottom=0.06, right=0.95, top=0.92, hspace=0.125, wspace=0.15)
fig.suptitle('Finetuning vs. Prompting/ICL vs. Instruction Tuning - MRPC', size=14)
fig.supxlabel('Train Dataset Size', size=14)
fig.supylabel('F1 Macro (+- STD)', size=14)

ax = fig.add_subplot(2, 2, 1)
mrpc_llama(mapping, ax)
ax = fig.add_subplot(2, 2, 2)
mrpc_roberta(mapping, ax)
ax = fig.add_subplot(2, 2, 3)
mrpc_bert(mapping, ax)
plt.savefig(os.path.join('visualisations', 'mrpc', f'mrpc_threshold_image_multi.png'), dpi=300, bbox_inches='tight')

plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, hspace=0.15, wspace=0.15)
fig.subplots_adjust(left=0.06, bottom=0.1, right=0.95, top=0.9, hspace=0.15, wspace=0.15)
fig.suptitle('Finetuning vs. Prompting/ICL vs. Instruction Tuning - BoolQ', size=14)
fig.supxlabel('Train Dataset Size', size=14)
fig.supylabel('F1 Macro (+- STD)', size=14)

ax = fig.add_subplot(1, 2, 1)
boolq_flanicl(mapping, ax)
ax = fig.add_subplot(1, 2, 2)
boolq_rest(mapping, ax)
plt.savefig(os.path.join('visualisations', 'boolq', f'boolq_threshold_image_multi.png'), dpi=300, bbox_inches='tight')