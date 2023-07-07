
import torch

import numpy as np
import pandas as pd
import os
import json
import pickle
import random

from sklearn.model_selection import train_test_split

from src.helpers import create_directory
from src.modeling.datasets.icl_dataset import ICLDataset
from src.modeling.models.in_context_learning import Alpaca


def split_train_validation_test(data, validation_split, test_split):
    validation_shape = int(validation_split * data.shape[0])
    test_shape = int(test_split * data.shape[0])
    train, test = train_test_split(data, test_size=test_shape, stratify=data['label'])
    train, validation = train_test_split(train, test_size=validation_shape, stratify=train['label'])
    #train, validation, test = np.split(
    #    data.sample(frac=1),
    #    [
    #        int((1 - test_split - validation_split) * data.shape[0]),
    #        int((1 - test_split) * data.shape[0])
    #    ]
    #)
    return train, validation, test


def split_train_test(data, test_split):
    train, test = train_test_split(data, test_size=int(test_split * data.shape[0]), stratify=data['label'])
    #train, test = np.split(
    #    data.sample(frac=1),
    #    [
    #        int((1 - test_split) * data.shape[0])
    #    ]
    #)
    return train, test


def select_labelled_subset(data, labels_to_use):
    #labelled, _ = train_test_split(data, train_size=labels_to_use, stratify=data['label'])
    labelled_true, _ = np.split(
        data[data['label'] == 0].sample(frac=1),
        [int(labels_to_use/2)]
    )
    labelled_false, _ = np.split(
        data[data['label'] == 1].sample(frac=1),
        [int(labels_to_use/2)]
    )
    labelled = pd.concat([labelled_true, labelled_false])
    #else:
    #    labelled, _ = np.split(
    #        data.sample(frac=1),
    #        [labels_to_use]
    #    )
    return labelled.sample(frac=1)


def save_results_json(results, path):
    file_name = f'run_{len(os.listdir(path))}.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(results, file)


def pickle_serialize(object_to_save, path):
    with open(path, 'wb') as file:
        pickle.dump(object_to_save, file)


def load_pickle_serialization(path):
    with open(path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def serialize_configuration(configuration, path):
    with open(path, 'w') as file:
        json.dump(configuration, file)


def load_serialized_configuration(path):
    with open(path, 'r') as file:
        configuration = json.load(file)
    return configuration


def select_data_to_use(train, validation, test, feature_to_use):
    train_X = train[feature_to_use].tolist()
    train_y = train['label'].to_numpy()

    validation_X = validation[feature_to_use].tolist()
    validation_y = validation['label'].to_numpy()

    test_X = test[feature_to_use].tolist()
    test_y = test['label'].to_numpy()

    return train_X, train_y, validation_X, validation_y, test_X, test_y
    # return train_X, train_y, test_X, test_y


def data_preprocess_factory(factor, mitigation_strategies, factor_path):
    factor_name = factor['name']
    data_split_strategy = factor['count'] if factor_name == 'data_split' else mitigation_strategies.get('data_split', 10)
    label_selection_strategy = factor['count'] if factor_name == 'label_selection' else mitigation_strategies.get('label_selection', 10)

    def __load_save_labelled__(data, test_split, labels_to_use, path):
        if os.path.exists(os.path.join(path, 'configuration.json')):
            print(f'Loading serialized labelled data!')
            configuration = load_serialized_configuration(os.path.join(path, 'configuration.json'))
            labelled = data.iloc[configuration['labelled']]
            test = None
        else:
            train, test = split_train_test(data, test_split)
            labelled = select_labelled_subset(train, labels_to_use)
            serialize_configuration(
                {
                    'labelled': labelled.index.tolist()
                },
                os.path.join(path, 'configuration.json')
            )
        return labelled, test
    
    def __load_save_split__(data, labelled, validation_split, test, path):
        if os.path.exists(os.path.join(path, 'configuration.json')):
            print(f'Loading serialized data split')
            configuration = load_serialized_configuration(os.path.join(path, 'configuration.json'))
            labelled_train = data.iloc[configuration['train']]
            labelled_validation = data.iloc[configuration['validation']]
            test = data.iloc[configuration['test']]
        else:
            labelled_train, labelled_validation = split_train_test(labelled, validation_split)
            serialize_configuration(
                {
                    'train': labelled_train.index.tolist(),
                    'validation': labelled_validation.index.tolist(),
                    'test': test.index.tolist()
                },
                os.path.join(path, 'configuration.json')
            )
        return labelled_train, labelled_validation, test

    def label_selection_investigation(data, validation_split, test_split, labels_to_use):
        for label_selection_idx in range(label_selection_strategy):
            label_selection_path = os.path.join(factor_path, f'label_{label_selection_idx + 1}')
            create_directory(label_selection_path)


            #labelled, test = __load_save_labelled__(data, test_split, labels_to_use, label_selection_path)
            if os.path.exists(os.path.join(label_selection_path, 'configuration.json')):
                print(f'Loading serialized data for labels at step {label_selection_idx + 1}')
                configuration = load_serialized_configuration(os.path.join(label_selection_path, 'configuration.json'))
                labelled = data.iloc[configuration['labelled']]
            else:
                train, test = split_train_test(data, test_split)
                labelled = select_labelled_subset(train, labels_to_use)
                serialize_configuration(
                    {
                        'labelled': labelled.index.tolist()
                    },
                    os.path.join(label_selection_path, 'configuration.json')
                )

            for data_split_idx in range(data_split_strategy):
                data_split_path = os.path.join(label_selection_path, f'split_{data_split_idx + 1}')
                create_directory(data_split_path)

                #labelled_train, labelled_validation, test = __load_save_split__(data, labelled, validation_split, test, data_split_path)
                if os.path.exists(os.path.join(data_split_path, 'configuration.json')):
                    print(f'Loading serialized data for split at step {data_split_idx}')
                    configuration = load_serialized_configuration(os.path.join(data_split_path, 'configuration.json'))
                    labelled_train = data.iloc[configuration['train']]
                    labelled_validation = data.iloc[configuration['validation']]
                    test = data.iloc[configuration['test']]
                else:
                    labelled_train, labelled_validation = split_train_test(labelled, validation_split)
                    serialize_configuration(
                        {
                            'train': labelled_train.index.tolist(),
                            'validation': labelled_validation.index.tolist(),
                            'test': test.index.tolist()
                        },
                        os.path.join(data_split_path, 'configuration.json')
                    )

                yield labelled_train, labelled_validation, test, data_split_path, data_split_idx, label_selection_idx
    
    def data_split_investigation(data, validation_split, test_split, labels_to_use):
        train_labels_to_use = int((1 - validation_split) * labels_to_use)
        validation_labels_to_use = int(validation_split * labels_to_use)
        for data_split_idx in range(data_split_strategy):
            data_split_path = os.path.join(factor_path, f'split_{data_split_idx + 1}')
            create_directory(data_split_path)

            if os.path.exists(os.path.join(data_split_path, 'configuration.json')):
                print(f'Loading serialized data for split at step {data_split_idx}')
                configuration = load_serialized_configuration(os.path.join(data_split_path, 'configuration.json'))
                train = data.iloc[configuration['train']]
                validation = data.iloc[configuration['validation']]
                test = data.iloc[configuration['test']]
            else:
                train, validation, test = split_train_validation_test(data, validation_split, test_split)
                serialize_configuration(
                    {
                        'train': train.index.tolist(),
                        'validation': validation.index.tolist(),
                        'test': test.index.tolist()
                    },
                    os.path.join(data_split_path, 'configuration.json')
                )

            for label_selection_idx in range(label_selection_strategy):
                label_selection_path = os.path.join(data_split_path, f'label_{label_selection_idx + 1}')
                create_directory(label_selection_path)

                if os.path.exists(os.path.join(label_selection_path, 'configuration.json')):
                    print(f'Loading serialized data for labels at step {label_selection_idx + 1}')
                    configuration = load_serialized_configuration(os.path.join(label_selection_path, 'configuration.json'))
                    labelled_train = data.iloc[configuration['train']]
                    labelled_validation = data.iloc[configuration['validation']]
                else:
                    labelled_train = select_labelled_subset(train, train_labels_to_use)
                    labelled_validation = select_labelled_subset(validation, validation_labels_to_use)
                    serialize_configuration(
                        {
                            'train': labelled_train.index.tolist(),
                            'validation': labelled_validation.index.tolist(),
                        },
                        os.path.join(label_selection_path, 'configuration.json')
                    )
                yield labelled_train, labelled_validation, test, label_selection_path, data_split_idx, label_selection_idx

    if factor_name == 'label_selection':
        return label_selection_investigation
    else:
        return data_split_investigation


def sample_choice_factory(factor, mitigation_strategies):
    factor_name = factor['name']
    sample_choice_strategy = factor['count'] if factor_name == 'sample_choice' else mitigation_strategies['sample_choice']
    
    def sample_choice_initialisation(train, validation, test, dataset_config, feature_to_use, out_path, device=None):
        train_X, train_y, validation_X, validation_y, test_X, test_y = select_data_to_use(train, validation, test, feature_to_use)

        if os.path.exists(os.path.join(out_path, 'sample_choice_seed_configuration.pkl')):
            seeds = load_pickle_serialization(os.path.join(out_path, 'sample_choice_seed_configuration.pkl'))
        else:
            seeds = [random.randint(1, 100000) for _ in range(sample_choice_strategy)]
            pickle_serialize(seeds, os.path.join(out_path, 'sample_choice_seed_configuration.pkl'))

        for sample_choice_idx, seed in enumerate(seeds):
            sample_choice_path = os.path.join(out_path, f'sample_choice_{str(sample_choice_idx + 1)}')
            create_directory(sample_choice_path)

            dataset = ICLDataset(
                train_data=(train_X, train_y),
                test_data=(test_X, test_y),
                num_shots=dataset_config['num_shots'],
                num_classes=dataset_config['num_classes'],
                choice_seed=seed,
                order_seed=None,
                device=device
            )

            dataset.choose_samples_for_prompt(seed)

            yield dataset, sample_choice_path, sample_choice_idx

    def default_choice_initialisation(train, validation, test, dataset_config, feature_to_use, out_path, device=None):
        train_X, train_y, validation_X, validation_y, test_X, test_y = select_data_to_use(train, validation, test, feature_to_use)

        dataset = ICLDataset(
            train_data=(train_X, train_y),
            test_data=(test_X, test_y),
            num_shots=dataset_config['num_shots'],
            num_classes=dataset_config['num_classes'],
            choice_seed=None,
            order_seed=None,
            device=device
        )

        for sample_choice_idx in range(sample_choice_strategy):
            sample_choice_path = os.path.join(out_path, f'sample_choice_{str(sample_choice_idx + 1)}')
            create_directory(sample_choice_path)
            dataset.choose_samples_for_prompt(None)
        
            yield dataset, sample_choice_path, sample_choice_idx

    if factor_name == 'sample_choice':
        return sample_choice_initialisation
    else:
        return default_choice_initialisation


def order_sample_factory(factor, mitigation_strategies):
    factor_name = factor['name']
    sample_order_strategy = factor['count'] if factor_name == 'sample_order' else mitigation_strategies['sample_order']

    def sample_order_initialisation(dataset, factor_path):
        if os.path.exists(os.path.join(factor_path, 'sample_order_seed_configuration.pkl')):
            seeds = load_pickle_serialization(os.path.join(factor_path, 'sample_order_seed_configuration.pkl'))
        else:
            seeds = [random.randint(1, 100000) for _ in range(sample_order_strategy)]
            pickle_serialize(seeds, os.path.join(factor_path, 'sample_order_seed_configuration.pkl'))

        for sample_order_number, seed in enumerate(seeds):
            sample_order_path = os.path.join(factor_path, f'order_{str(sample_order_number + 1)}')
            create_directory(sample_order_path)

            dataset.reorder_samples_for_prompt(seed)

            yield dataset, sample_order_path, sample_order_number
    
    def default_sample_order_initialisation(dataset, factor_path):
        for sample_order_number in range(sample_order_strategy):
            sample_order_path = os.path.join(factor_path, f'order_{str(sample_order_number + 1)}')
            create_directory(sample_order_path)

            dataset.reorder_samples_for_prompt(None)

            yield dataset, sample_order_path, sample_order_number

    if factor_name == 'sample_order':
        return sample_order_initialisation
    else:
        return default_sample_order_initialisation


def run_in_context_learning(config, data, labels, output_path):
    if config['modeling'].get('investigation_type', '') == 'golden_model':
        factor = {'name': 'golden_model'}
        run_investigation_with_mitigation(factor, config, data, labels, output_path)
    elif config['modeling'].get('investigation_type', '') == 'factors':
        for factor in config['modeling'].get('factors', []):
            print(f'Running factor {factor.get("name")}')
            run_investigation_with_mitigation(factor, config, data, labels, output_path)
    else:
        pass
    

def run_investigation_with_mitigation(factor, config, data, labels, output_path):
    factor_name = factor['name']
    factor_path = os.path.join(output_path, 'predictions', factor_name)

    modeling_config = config['modeling']
    dataset_config = modeling_config['dataset_parameters']
    mitigation_strategies = modeling_config['mitigations_strategies']

    device = torch.device('cuda:0' if torch.cuda.is_available() and modeling_config['use_gpu'] else 'cpu')

    data['label'] = labels
    test_split = modeling_config['train_test_split']
    validation_split = modeling_config['train_valid_split']
    feature_to_use = modeling_config['feature']
    labels_to_use = modeling_config['labels_to_use']

    model = Alpaca(device)

    data_factory = data_preprocess_factory(factor, mitigation_strategies, factor_path)
    sample_factory = sample_choice_factory(factor, mitigation_strategies)
    order_factory = order_sample_factory(factor, mitigation_strategies)
    # dataset_factory = dataset_initialisation_factory(factor, mitigation_strategies, factor_path)
    # run_factory = run_initialisation_factory(factor, mitigation_strategies, factor_path, modeling_config)

    for labelled_train, labelled_validation, test, out_path, data_split_number, label_selection_number in data_factory(data, validation_split, test_split, labels_to_use):
        for dataset, choice_path, choice_number in sample_factory(labelled_train, labelled_validation, test, dataset_config, feature_to_use, out_path, device):
            for dataset, order_path, order_number in order_factory(dataset, choice_path):

                dataset.prepare_prompt_for_data(dataset_config['instructions'])
                results_path = order_path
                if not os.path.exists(os.path.join(results_path, 'results.json')):
                    print(f'Evaluating model!')
                    # predictions = model.evaluate_in_batch(dataset, modeling_config['evaluation_batch'])
                    predictions = model.evaluate(dataset)
                    with open(os.path.join(results_path, 'results.json'), 'w') as file:
                        json.dump(
                            {
                                'split_number': data_split_number + 1,
                                'label_selection': label_selection_number + 1,
                                'sample_choice': choice_number + 1,
                                'sample_order': order_number,
                                'base_model': 'Alpaca-7B',
                                'predictions': predictions
                            },
                            file
                        )
