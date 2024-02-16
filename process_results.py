import pickle
import os
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

DATASET_PATH = 'snips'
MITIGATION_RUNS = 100
INVESTIGATION_RUNS = 10
TEST_DATA_FRACTION = 1
SEED = 1337
MODEL = 'flan-t5'
L3D = 'icl'

SUBSELECT_DATA = TEST_DATA_FRACTION < 1

if SUBSELECT_DATA:
    path = os.path.join('results', 'investigation_{L3D}_{MODEL}_base', 'stability', DATASET_PATH, 'golden_model', f'mitigation_0', 'investigation_0', 'results.json')
    with open(path, 'r') as file:
        data = json.load(file)
    indices = np.arange(len(data['real']))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    to_select = 500
    indices = indices[:to_select]


classes_mapper = {
    'sst2': 2,
    'cola': 2,
    'mrpc': 2,
    'trec': 6,
    'ag_news': 4,
    'snips': 7,
    'db_pedia': 14,
}


csv_results = []

for DATASET_PATH in ['sst2', 'cola', 'mrpc', 'ag_news', 'trec', 'snips', 'db_pedia']:
    print(f'-------- {DATASET_PATH} --------')
    for MODEL in ['flan-t5', 'llama2', 'mistral', 'zephyr', 'bert', 'roberta']:
        print(f'-------- {MODEL} --------')
        num_classes = classes_mapper[DATASET_PATH]
        results = {
            'data_split': {},
            'label_choice': {},
            'sample_choice': {},
            'sample_order': {},
            'model_initialisation': {},
        }

        if MODEL in ['bert', 'roberta']:
            L3D = 'finetuning'
        else:
            L3D = 'icl'
        MITIGATION_RUNS = 100
        INVESTIGATION_RUNS = 10
        base_path = os.path.join('results', 'stability', f'{L3D}_{MODEL}_base', 'prompt_0', DATASET_PATH)
            

        golden_model = []

        all = failed = 0
        for mitigation in range(MITIGATION_RUNS * INVESTIGATION_RUNS):
            path = os.path.join(base_path, 'golden_model', f'mitigation_{mitigation}', 'investigation_0', 'results.json')
            
            try:
                with open(path, 'r') as file:
                    data = json.load(file)
                if SUBSELECT_DATA:
                    score = f1_score(np.array(data['real'])[indices], np.array(data['predicted'])[indices], average='macro')
                else:
                    score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
                if score < 1.0/num_classes:
                    failed += 1
                all += 1
                golden_model.append(score * 100)
            except Exception as e:
                # print(e)
                continue

        print(f'Golden model:')
        print(f'Failed percentage: {failed / all * 100}')
        print(f'mean: {np.mean(golden_model)}')
        print(f'std: {np.std(golden_model)}')
        print(f'Min: {np.min(golden_model)}')
        print(f'Max: {np.max(golden_model)}')
        performance = np.mean(golden_model)
        gm_std = np.std(golden_model)
        new_result = {
            'model': MODEL,
            'dataset': DATASET_PATH,
            'factor': 'golden_model',
            'performance': performance,
            'deviation': gm_std,
            'c_std': 0,
            'm_std': 0,
            'importance_gm': 0,
            'importance_factor': 0,
            'failed': failed,
            'all': all,
        }
        csv_results.append(new_result)
        print()

        factors = ['label_choice', 'data_split', 'sample_order', 'sample_choice'] if L3D == 'icl' else ['label_choice', 'data_split', 'sample_order', 'model_initialisation']
        for factor in factors:
            all = failed = 0
            results[factor]['full'] = []
            results[factor]['agg'] = {}
            for mitigation in range(MITIGATION_RUNS):
                results[factor]['agg'][mitigation] = []
                for investigation in range(INVESTIGATION_RUNS):
                    path = os.path.join(base_path, factor, f'mitigation_{mitigation}', f'investigation_{investigation}', 'results.json')
                    try:
                        with open(path, 'r') as file:
                            data = json.load(file)

                        if SUBSELECT_DATA:
                            score = f1_score(np.array(data['real'])[indices], np.array(data['predicted'])[indices], average='macro')
                        else:
                            score = f1_score(np.array(data['real']), np.array(data['predicted']), average='macro')
                        if score < 1.0/num_classes:
                            failed += 1
                        all += 1
                        results[factor]['agg'][mitigation].append(score * 100)
                        results[factor]['full'].append(score * 100)
                    except Exception as e:
                        # print(e)
                        continue
            results[factor]['all'] = all
            results[factor]['failed'] = failed
            print(factor)
            print(f'Failed percentage: {results[factor]["failed"] / results[factor]["all"] * 100}')
            print(f'Mean: {np.mean(results[factor]["full"])}, Std: {np.std(results[factor]["full"])}, Min: {np.min(results[factor]["full"])}, Max: {np.max(results[factor]["full"])}')
            run_std = np.std(results[factor]["full"])
            investigated = [np.std(val) for val in results[factor]['agg'].values()]
            mitigated = [np.mean(val) for val in results[factor]['agg'].values()]
            print(f'Contributed std: {np.mean(investigated)}')
            print(f'Mitigated std: {np.std(mitigated)}')
            print(f'Importance: {((np.mean(investigated) - np.std(mitigated))/gm_std)}')
            print(f'Importance (second): {((np.mean(investigated) - np.std(mitigated))/run_std)}')
            performance = np.mean(results[factor]["full"])
            c_std = np.mean(investigated)
            m_std = np.std(mitigated)
            importance_gm = (np.mean(investigated) - np.std(mitigated))/gm_std
            importance_factor = (np.mean(investigated) - np.std(mitigated))/run_std
            new_result = {
                'model': MODEL,
                'dataset': DATASET_PATH,
                'factor': factor,
                'performance': performance,
                'deviation': run_std,
                'c_std': c_std,
                'm_std': m_std,
                'importance_gm': importance_gm,
                'importance_factor': importance_factor,
                'failed': results[factor]["failed"],
                'all': results[factor]["all"],
            }
            csv_results.append(new_result)
        print()

    

df = pd.DataFrame(csv_results)
print(df)
df.to_csv('full_results.csv')