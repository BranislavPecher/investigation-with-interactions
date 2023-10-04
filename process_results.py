import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score

DATASET_PATH = 'mrpc'
MITIGATION_RUNS = 10
INVESTIGATION_RUNS = 10
TEST_DATA_FRACTION = 1
SEED = 1337
MODEL = 'llama2' # 'llama2'
L3D = 'icl' # 'icl'

SUBSELECT_DATA = TEST_DATA_FRACTION < 1

if SUBSELECT_DATA:
    path = os.path.join('results', 'investigation_{L3D}_{MODEL}_base', 'stability', DATASET_PATH, 'golden_model', f'mitigation_0', 'investigation_0', 'results.json')
    with open(path, 'r') as file:
        data = json.load(file)
    indices = np.arange(len(data['real']))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    # to_select = int(indices.shape[0] * TEST_DATA_FRACTION)
    to_select = 500
    indices = indices[:to_select]


results = {
    'data_split': {},
    'label_choice': {},
    'sample_choice': {},
    'sample_order': {},
    'model_initialisation': {},
}

golden_model = []

all = failed = 0
for mitigation in range(MITIGATION_RUNS * INVESTIGATION_RUNS):
    path = os.path.join('results', f'investigation_{L3D}_{MODEL}_base', 'stability', DATASET_PATH, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')

    with open(path, 'r') as file:
        data = json.load(file)
    if SUBSELECT_DATA:
        score = f1_score(np.array(data['real'])[indices], np.array(data['predicted'])[indices], average='macro')
    else:
        score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
    if score < .5:
        failed += 1
    all += 1
    golden_model.append(score * 100)

print(f'Golden model:')
print(f'Failed percentage: {failed / all * 100}')
print(f'Mean: {np.mean(golden_model)}')
print(f'Std: {np.std(golden_model)}')
print(f'Min: {np.min(golden_model)}')
print(f'Max: {np.max(golden_model)}')
print()

for factor in ['data_split', 'label_choice', 'sample_choice', 'sample_order']:
# for factor in ['data_split', 'label_choice', 'model_initialisation', 'sample_order']:
    all = failed = 0
    results[factor]['full'] = []
    results[factor]['agg'] = {}
    for mitigation in range(MITIGATION_RUNS):
        results[factor]['agg'][mitigation] = []
        for investigation in range(INVESTIGATION_RUNS):
            path = os.path.join('results', f'investigation_{L3D}_{MODEL}_base', 'stability', DATASET_PATH, factor, f'mitigation_{mitigation}', f'investigation_{investigation}', 'results.json')

            with open(path, 'r') as file:
                data = json.load(file)

            if SUBSELECT_DATA:
                score = f1_score(np.array(data['real'])[indices], np.array(data['predicted'])[indices], average='macro')
            else:
                score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
            if score < .5:
                failed += 1
            all += 1
            results[factor]['agg'][mitigation].append(score * 100)
            results[factor]['full'].append(score * 100)
    results[factor]['all'] = all
    results[factor]['failed'] = failed
    print(factor)
    print(f'Failed percentage: {results[factor]["failed"] / results[factor]["all"] * 100}')
    print(f'Mean: {np.mean(results[factor]["full"])}, Std: {np.std(results[factor]["full"])}, Min: {np.min(results[factor]["full"])}, Max: {np.max(results[factor]["full"])}')
    investigated = [np.std(val) for val in results[factor]['agg'].values()]
    mitigated = [np.mean(val) for val in results[factor]['agg'].values()]
    print(f'Contributed std: {np.mean(investigated)}')
    print(f'Mitigated std: {np.std(mitigated)}')
    print()

    
