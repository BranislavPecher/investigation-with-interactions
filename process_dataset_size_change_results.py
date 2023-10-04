import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
from data import TextDataset

DATASET_PATH = 'sst2'
# DATASET_PATH = 'mrpc'
# DATASET_PATH = 'boolq'

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


def load_random_baseline(sizes, runs=100):
    scores = []
    for mitigation in range(runs):
        path = os.path.join('results', f'dataset_size_change_prompting_flan-t5_base', f'num_samples_1000', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            # score = np.max([f1_score(np.zeros(len(data['predicted'])), np.array(data['real']), average='macro'), f1_score(np.ones(len(data['predicted'])), np.array(data['real']), average='macro')])
            score = np.max([np.sum(np.array(data['predicted']) == 1) / np.array(data['predicted']).shape[0], np.sum(np.array(data['predicted']) == 0) / np.array(data['predicted']).shape[0]])
            scores.append(score * 100)
        except:
            continue
    return np.repeat(np.array([np.mean(scores)]), len(sizes))


plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(16, 5))

LOCATION = 'lower'
if LOCATION == 'upper':
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.8, hspace=0.15, wspace=0.08)
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.8, hspace=0.15, wspace=0.08)
elif LOCATION == 'lower':
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.89, hspace=0.15, wspace=0.08)
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.89, hspace=0.15, wspace=0.08)
fig.suptitle('Finetuning vs. Prompting/ICL vs. Instruction Tuning', size=14)
fig.supxlabel('Train Dataset Size (log)', size=14)
fig.supylabel('F1 Macro (+- STD)', size=14)
for config_idx, meta_config in enumerate([
    ('sst2', 'SST2', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000)),
    ('mrpc', 'MRPC', (10, 50, 100, 250, 500, 1000, 2500, 5000)),
    ('boolq', 'BoolQ', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)),
    ]):
    DATASET_PATH, dataset_name, sizes = meta_config
    
    ax = fig.add_subplot(1, 3, config_idx + 1)
    configurations = [('bert', 'finetuning'), ('roberta', 'finetuning'), ('flan-t5', 'prompting'), ('flan-t5', 'icl'), ('llama2', 'prompting'), ('llama2', 'icl'), ('chatgpt', 'prompting'), ('chatgpt', 'icl'), ('flan-t5', 'instruction_tuning_steps')]
    models = ['bert', 'roberta', 'flan-t5', 'llama2', 'chatgpt', 'flan-t5-instruction', 'random']
    clrs = sns.color_palette("deep", len(models))

    configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
    color_idx = 0
    for idx, config in enumerate(configurations):
        model_name, l3d_type, display_name = config
        print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
        means, stds = load_results(sizes, l3d_type, model_name, 100, None)
        with sns.axes_style("darkgrid"):
            ax.plot(sizes, means, label=display_name, color=clrs[color_idx])
            ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=clrs[color_idx])
        color_idx += 1

    for model_name in ['flan-t5']:
        for l3d_type in ['instruction_tuning_steps']:
            for to_load, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                with sns.axes_style("darkgrid"):
                    ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=clrs[color_idx], linestyle='dashed')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

    for model_name in ['flan-t5']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None)
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=clrs[color_idx], linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else 0.1, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

    for model_name in ['llama2']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 100, None if l3d_type == 'icl' else 1000)
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=clrs[color_idx], linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

    for model_name in ['chatgpt']:
        for l3d_type, display_name in [('prompting', 'Prompting'), ('icl', 'ICL')]:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
            with sns.axes_style("darkgrid"):
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=clrs[color_idx], linestyle='dashed')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.1, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

    random_baseline = load_random_baseline(sizes)
    print(f'Random Baseline: {random_baseline[0]}')
    with sns.axes_style("darkgrid"):
        epochs = list(range(len(sizes)))
        ax.plot(sizes, random_baseline, label='Random Baseline', color=clrs[color_idx], linestyle='dotted')

    ax.set_title(dataset_name)
    ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc=f'{LOCATION} center', ncols=6, fontsize=12, bbox_to_anchor=(0.5, 0.95 if LOCATION == 'upper' else -0.125))
plt.savefig(os.path.join('visualisations', f'meta_image_multi_new_{LOCATION}_improved.png'), dpi=600, bbox_inches='tight')