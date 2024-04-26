import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
from data import TextDataset
import colorcet as cc
import matplotlib.gridspec as gridspec

# DATASET_PATH = 'sst2'
# DATASET_PATH = 'mrpc'
DATASET_PATH = 'boolq'

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_results(sizes, l3d_type, model_name, runs=100, special_type=None, to_load=None):
    means = []
    stds = []
    if special_type is not None:
        special_type_idx = 0
    for size in sizes:
        golden_model = []
        all = failed = 0
        for mitigation in range(runs):
            if special_type is None:
                path = os.path.join('results', 'dataset_size_change', f'{l3d_type}_{model_name}_base', f'num_labelled_{size}', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
            else:
                if type(special_type) == int:
                    path = os.path.join('results', 'dataset_size_change', f'{l3d_type}_{model_name}_base', f'num_labelled_{special_type}', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
                elif type(special_type) == list:
                    if (special_type_idx + 1) < len(special_type) and size >= special_type[special_type_idx + 1]:
                        special_type_idx += 1
                    path = os.path.join('results', 'dataset_size_change', f'{l3d_type}_{model_name}_base', f'num_labelled_{special_type[special_type_idx]}', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
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
                # print(e)
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
        path = os.path.join('results', f'dataset_size_change', 'prompting_flan-t5_base', f'num_labelled_1000', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
        try:
            with open(path, 'r') as file:
                data = json.load(file)
            score = np.max([np.sum(np.array(data['real']) == 1) / np.array(data['real']).shape[0], np.sum(np.array(data['real']) == 0) / np.array(data['real']).shape[0]])
            scores.append(score * 100)
        except:
            continue
    return np.repeat(np.array([np.mean(scores)]), len(sizes))


plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(20, 14))

outer = gridspec.GridSpec(4, 2)

LOCATION = 'lower'

if LOCATION == 'upper':
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.8, hspace=0.15, wspace=0.08)
    fig.subplots_adjust(left=0.06, bottom=0.1, right=0.98, top=0.8, hspace=0.15, wspace=0.08)
elif LOCATION == 'lower':
    plt.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.94, hspace=0.2, wspace=0.065)
    fig.subplots_adjust(left=0.05, bottom=0.04, right=0.98, top=0.94, hspace=0.2, wspace=0.065)
    
fig.suptitle(f'Finetuning vs. Prompting/ICL vs. Instruction Tuning - Binary and Multiclass Datasets', size=16)
fig.supxlabel('Train Dataset Size (log)', size=14)
fig.supylabel('F1 Macro (+- STD)', size=14)
alpha = 0.05

datasets_and_sizes = [
    ('sst2', 'SST2', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000)),
    ('mrpc', 'MRPC', (10, 50, 100, 250, 500, 1000, 2500, 5000)),
    ('cola', 'CoLA', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)),
    ('boolq', 'BoolQ', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)),
    ('ag_news', 'AG News', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)),
    ('trec', 'TREC', (10, 50, 100, 250, 500, 1000, 2500, 5000)),
    ('snips', 'SNIPS', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000)),
    ('db_pedia', 'DB Pedia', (10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000)),
]


    
for config_idx, meta_config in enumerate(datasets_and_sizes):
    DATASET_PATH, dataset_name, sizes = meta_config
    special_type = None if DATASET_PATH in ['sst2', 'mrpc', 'boolq'] else [50, 500, 1000, 2500, 5000]
    number = 100 if DATASET_PATH in ['sst2', 'mrpc', 'boolq'] else 10

    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[config_idx], wspace=0)
    old_ax = None
    for l3d_type_idx, conf in enumerate([('prompting', 'Prompting'), ('icl', 'ICL')]):
        if l3d_type_idx == 0:
            ax = fig.add_subplot(gs[0, l3d_type_idx])
            old_ax = ax
        else:
            ax = fig.add_subplot(gs[0, l3d_type_idx])
            ax.set(yticklabels=[])
            ax.set(ylabel=None)
            ax.tick_params(left=False)

        configurations = [('bert', 'finetuning'), ('roberta', 'finetuning'), ('flan-t5', 'prompting'), ('flan-t5', 'icl'), ('llama2', 'prompting'), ('llama2', 'icl'), ('chatgpt', 'prompting'), ('chatgpt', 'icl'), ('flan-t5', 'instruction_tuning_steps'), ('mistral', 'prompting'), ('mistral', 'icl'), ('zephyr', 'prompting'), ('zephyr', 'icl'), ('mistral', 'instruction_tuning_steps')]
        models = ['bert', 'roberta', 'flan-t5', 'llama2', 'chatgpt',  'mistral', 'zephyr', 'flan-t5-instruction', 'mistral-instruction', 'zephyr-instruction', 'random']
        clrs = sns.color_palette("deep", len(models))

        configurations = [('bert', 'finetuning', 'BERT FineTuning'), ('roberta', 'finetuning', 'RoBERTa FineTuning')]
        color_idx = 0
        for idx, config in enumerate(configurations):
            model_name, l3d_type, display_name = config
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, 100, None)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx])
                ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx])
            color_idx += 1

        to_load, display_name = conf
        for model_name in ['flan-t5']:
            for l3d_type in ['instruction_tuning_steps']:
                display_name = f'Flan-T5 Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl')):
                    means, stds = load_results(sizes, l3d_type, model_name, 100, None, to_load)
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'wb') as file:
                        pickle.dump({'means': means, 'stds': stds}, file)
                else:
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'rb') as file:
                        pickled = pickle.load(file)
                        means = pickled['means']
                        stds = pickled['stds']
                with sns.axes_style("darkgrid"):
                    epochs = list(range(len(sizes)))
                    ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='\\')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='solid')
            color_idx += 1

        to_load, display_name = conf
        for model_name in ['mistral']:
            for l3d_type in ['instruction_tuning_steps']:
                display_name = f'Mistral-7B Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl')):
                    means, stds = load_results(sizes, l3d_type, model_name, 10, None, to_load)
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'wb') as file:
                        pickle.dump({'means': means, 'stds': stds}, file)
                else:
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'rb') as file:
                        pickled = pickle.load(file)
                        means = pickled['means']
                        stds = pickled['stds']

                with sns.axes_style("darkgrid"):
                    epochs = list(range(len(sizes)))
                    ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='\\')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='solid')
            color_idx += 1

        to_load, display_name = conf
        for model_name in ['zephyr']:
            for l3d_type in ['instruction_tuning_steps']:
                display_name = f'Zephyr-7B Instruction Tuning - {display_name}'
                print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
                if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl')):
                    means, stds = load_results(sizes, l3d_type, model_name, 10, None, to_load)
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'wb') as file:
                        pickle.dump({'means': means, 'stds': stds}, file)
                else:
                    with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}_{to_load}.pkl'), 'rb') as file:
                        pickled = pickle.load(file)
                        means = pickled['means']
                        stds = pickled['stds']

                with sns.axes_style("darkgrid"):
                    epochs = list(range(len(sizes)))
                    ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if to_load == 'icl' else 'solid'))
                    if to_load == 'icl':
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='\\')
                    else:
                        ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='solid')
            color_idx += 1

        l3d_type, display_name = conf
        for model_name in ['flan-t5']:
            display_name = f'Flan-T5 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, number, special_type if l3d_type == 'icl' else 1000)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                if DATASET_PATH == 'snips':
                    print(means)
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='\\')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=0.05 if DATASET_PATH == 'boolq' else alpha, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

        l3d_type, display_name = conf
        for model_name in ['llama2']:
            display_name = f'LLaMA-2 {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, number, special_type if l3d_type == 'icl' else 1000)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='-')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

        l3d_type, display_name = conf
        for model_name in ['chatgpt']:
            display_name = f'ChatGPT {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, 6, 1000)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='/')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

        l3d_type, display_name = conf
        for model_name in ['mistral']:
            display_name = f'Mistral-7B {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, number, [50, 500, 1000, 2500, 5000] if l3d_type == 'icl' else 1000)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='/')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

        l3d_type, display_name = conf
        for model_name in ['zephyr']:
            display_name = f'Zephyr-7B {display_name}'
            print(f'Model: {model_name.capitalize()}, L3D type: {l3d_type}')
            if not os.path.exists(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl')):
                means, stds = load_results(sizes, l3d_type, model_name, number, [50, 500, 1000, 2500, 5000] if l3d_type == 'icl' else 1000)
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'wb') as file:
                    pickle.dump({'means': means, 'stds': stds}, file)
            else:
                with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_{model_name}_{l3d_type}.pkl'), 'rb') as file:
                    pickled = pickle.load(file)
                    means = pickled['means']
                    stds = pickled['stds']
            with sns.axes_style("darkgrid"):
                epochs = list(range(len(sizes)))
                ax.plot(sizes, means, label=display_name, color=clrs[color_idx], linestyle=('dashed' if l3d_type == 'icl' else 'solid'))
                if l3d_type == 'icl':
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='dashed')#, hatch='/')
                else:
                    ax.fill_between(sizes, means-stds, means+stds, alpha=alpha, color=clrs[color_idx], linestyle='solid')
        color_idx += 1

        random_baseline = load_random_baseline(sizes)
        print(f'Random Baseline: {random_baseline[0]}')
        with sns.axes_style("darkgrid"):
            epochs = list(range(len(sizes)))
            ax.plot(sizes, random_baseline, label='Random Baseline', color='black', linestyle='dotted')
        with open(os.path.join('pickled', 'dataset_size_change', 'full', f'{DATASET_PATH}_random_baseline.pkl'), 'wb') as file:
            pickle.dump({'means': random_baseline}, file)

        ax.set_title(f'{dataset_name} ({conf[1]})')
        ax.set_xscale('log')
        if DATASET_PATH == 'cola':
            ax.set_ylim(bottom=18)
        if DATASET_PATH == 'boolq':
            ax.set_ylim(top=85)
        if DATASET_PATH == 'sst2':
            ax.set_ylim(top=96)
        if DATASET_PATH == 'db_pedia':
            ax.set_ylim(bottom=-2)
        
        if config_idx == 0:
            if l3d_type_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
            elif l3d_type_idx == 1:
                handles_t, labels_t = ax.get_legend_handles_labels()
                handles.extend(handles_t)
                labels.extend(labels_t)

                indices = [0, 1, 2, 13, 3, 14, 4, 15, 5, 16, 6, 17, 7, 18, 8, 19, 9, 20, 10]
                handles = [handles[i] for i in indices]
                labels = [labels[i] for i in indices]

    fig.legend(handles, labels, loc=f'{LOCATION} center', ncols=5, fontsize=12, bbox_to_anchor=(0.5, 0.95 if LOCATION == 'upper' else -0.075))
    plt.savefig(os.path.join('visualisations', 'size_change', f'meta-full-image.png'), dpi=300, bbox_inches='tight')