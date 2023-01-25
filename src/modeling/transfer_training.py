from distutils.command.config import config
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
import json
import copy
import pickle
import random
import re

from sklearn.model_selection import ParameterGrid, train_test_split

from src.helpers import create_directory
from src.modeling.datasets.meta_text_dataset import MetaTextDataset
from src.modeling.models.base_models import MetaSimpleCnnText, DenseClassifier
from src.modeling.models.transfer_learning import BERT

base_model_mapping = {
    'MetaSimpleCnnText': MetaSimpleCnnText,
    'DenseClassifier': DenseClassifier
}

transfer_learning_model_mapping = {
    'BERT': BERT
}

transfer_learning_model_mapping = {
    'BERT': BERT,
}

optimizer_mapping = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

loss_mapping = {
    'cross_entropy': F.cross_entropy
}

scheduler_mapping = {
    'base': torch.optim.lr_scheduler.StepLR
}

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


def select_adaptation_data(data, labels, num_shots, seed=None):
    true_labels = [idx for idx, label in enumerate(labels) if label == 1]
    false_labels = [idx for idx, label in enumerate(labels) if label == 0]
    if seed is not None:
        random.seed(seed)
    indices = random.sample(true_labels, num_shots) + random.sample(false_labels, num_shots)
    random.shuffle(indices)
    adaptation_data = [data[idx] for idx in indices]
    adaptation_labels = labels[indices]
    return adaptation_data, adaptation_labels


def initialize_transfer_learner_model(base_model_config, transfer_model_config, dataset_config, modeling_config):
    model_params = base_model_config['params']
    net = base_model_mapping[base_model_config['model']](
        dataset_config['max_sentence_length'],
        dataset_config['embedding_dimension'],
        model_params['n_filters'],
        model_params['filter_sizes'],
        model_params['pool_size'],
        model_params['hidden_size'],
        dataset_config['num_classes'],
        model_params.get('initialise_classifier', True)
    )
    loss_function = loss_mapping[model_params['loss_function']]
    device = torch.device('cuda:0' if torch.cuda.is_available() and modeling_config['use_gpu'] else 'cpu')
    params = parse_transfer_learner_config(modeling_config['transfer_learning_parameters'], transfer_model_config['params'])
    params['optimizer'] = optimizer_mapping[params['optimizer']] if params.get('optimizer') else None
    params['loss_function'] = loss_function
    transfer_model = transfer_learning_model_mapping[transfer_model_config['model']](
        model=net,
        params=params,
        device=device
    )
    return transfer_model, params


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


def parse_transfer_learner_config(basic_config, advanced_config):
    config_params = {
        'num_epochs': None,
        'lr': None,
        'optimizer': None,
        'evaluation_batch': None,
        'batch_size': None
    }

    return __parse_config__(basic_config, advanced_config, config_params)


def parse_dataset_config(basic_config, advanced_config):
    config_params = {
        'max_sentence_length': None,
        'embedding_dimension': None,
        'num_classes': None,
        'initialise_classifier': None
    }

    return __parse_config__(basic_config, advanced_config, config_params)


def __parse_config__(basic_config, advanced_config, config_to_fill):
    for key in config_to_fill.keys():
        config_to_fill[key] = advanced_config[key] if advanced_config.get(key) is not None else basic_config.get(key)
    return config_to_fill


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


def dataset_initialisation_factory(factor, mitigation_strategies, factor_path):
    factor_name = factor['name']
    data_order_strategy = factor['count'] if factor_name == 'data_order' else 1
    
    def data_order_dataset_initialisation(train, validation, test, dataset_config, feature_to_use, out_path, device=None):
        train_X, train_y, validation_X, validation_y, test_X, test_y = select_data_to_use(train, validation, test, feature_to_use)

        if os.path.exists(os.path.join(factor_path, 'seed_configuration.pkl')) and os.path.exists(os.path.join(factor_path, 'seeds_for_fine_tuning_order.pkl')):
            seeds = load_pickle_serialization(os.path.join(factor_path, 'seeds_for_fine_tuning_order.pkl'))
        else:
            if os.path.exists(os.path.join(factor_path, 'seed_configuration.pkl')):
                default_seeds = load_pickle_serialization(os.path.join(factor_path, 'seed_configuration.pkl'))
                if len(default_seeds) < data_order_strategy:
                    default_seeds.extend([random.randint(1, 100000) for _ in range(data_order_strategy - len(default_seeds))])
                    pickle_serialize(default_seeds, os.path.join(factor_path, 'seed_configuration.pkl'))
            else:
                default_seeds = [random.randint(1, 100000) for _ in range(data_order_strategy)]
            seeds = []
            for seed in default_seeds:
                random.seed(seed)
                seeds.append([random.randint(1, 100000) for _ in range(100)])

            pickle_serialize(seeds, os.path.join(factor_path, 'seeds_for_fine_tuning_order.pkl'))

        for data_order_idx, seed in enumerate(seeds):
            data_order_path = os.path.join(out_path, f'data_order_{str(data_order_idx + 1)}')
            create_directory(data_order_path)

            train_dataset = MetaTextDataset(
                train_data=(train_X, train_y),
                test_data=(validation_X, validation_y),
                max_length=dataset_config['max_sentence_length'],
                preembed_data=True,
                task_definition='from_seed',
                transfer_learning=True,
                seed=seed,
                device=device
            )

            test_dataset = MetaTextDataset(
                train_data=(train_X, train_y),
                test_data=(test_X, test_y),
                max_length=dataset_config['max_sentence_length'],
                transfer_learning=True,
                preembed_data=True,
                device=device
            )


            yield train_dataset, test_dataset, data_order_path, data_order_idx



    def default_dataset_initialisation(train, validation, test, dataset_config, feature_to_use, out_path, device=None):
        train_X, train_y, validation_X, validation_y, test_X, test_y = select_data_to_use(train, validation, test, feature_to_use)

        train_dataset = MetaTextDataset(
            train_data=(train_X, train_y),
            test_data=(validation_X, validation_y),
            max_length=dataset_config['max_sentence_length'],
            transfer_learning=True,
            preembed_data=True,
            device=device
        )

        test_dataset = MetaTextDataset(
            train_data=(train_X, train_y),
            test_data=(test_X, test_y),
            max_length=dataset_config['max_sentence_length'],
            transfer_learning=True,
            preembed_data=True,
            device=device
        )
        yield train_dataset, test_dataset, out_path, 0

    if factor_name == 'data_order':
        return data_order_dataset_initialisation
    else:
        return default_dataset_initialisation


def run_initialisation_factory(factor, mitigation_strategies, factor_path, configuration):
    factor_name = factor['name']
    run_number_strategy = mitigation_strategies['repeated_runs']
    model_initialisation_strategy = factor['count'] if factor_name == 'model_initialisation' else 1
    modeling_config = configuration
    dataset_config = modeling_config['dataset_parameters']
    transfer_models = {}

    def initialize_general_meta_learner_model(modeling_config, run_path):
        for run_number in range(run_number_strategy):
            run_number_path = os.path.join(run_path, f'run_{str(run_number + 1)}')
            create_directory(run_number_path)

            for transfer_model_config in modeling_config['transfer_models']:
                transfer_model_name = transfer_model_config['model']
                for base_model_config in modeling_config['base_models']:
                    base_model_name = base_model_config['model']
                    transfer_learner_path = os.path.join(run_number_path, transfer_model_config['model'])
                    create_directory(transfer_learner_path)
                    initialized_transfer_model, params = initialize_transfer_learner_model(base_model_config, transfer_model_config, dataset_config, modeling_config)
                    yield initialized_transfer_model, params, transfer_learner_path, run_number, transfer_model_name, base_model_name
    
    def initialize_meta_learner_model_for_multiple_runs(modeling_config, run_path):
        for initialisation_number in range(model_initialisation_strategy):
            init_path = os.path.join(run_path, f'init_{initialisation_number + 1}')
            create_directory(init_path)
            for run_number in range(run_number_strategy):
                run_number_path = os.path.join(init_path, f'run_{str(run_number + 1)}')
                create_directory(run_number_path)
                for transfer_model_config in modeling_config['transfer_models']:
                    transfer_model_name = transfer_model_config['model']
                    for base_model_config in modeling_config['base_models']:
                        base_model_name = base_model_config['model']
                        transfer_model, params = transfer_models[transfer_model_name][base_model_name][initialisation_number]
                        transfer_learner_path = os.path.join(run_number_path, transfer_model_name)
                        create_directory(transfer_learner_path)
                        yield copy.deepcopy(transfer_model), params, transfer_learner_path, run_number, transfer_model_name, base_model_name
    
    if factor_name == 'model_initialisation':
        for transfer_model_config in modeling_config['transfer_models']:
            transfer_model_name = transfer_model_config['model']
            transfer_model_path = os.path.join(factor_path, transfer_model_name)
            if transfer_models.get(transfer_model_name) is None:
                transfer_models[transfer_model_name] = {}
                create_directory(transfer_model_path)
            for base_model_config in modeling_config['base_models']:
                base_model_name = base_model_config['model']
                base_model_path = os.path.join(transfer_model_path, base_model_name)
                create_directory(base_model_path)
                transfer_models[transfer_model_name][base_model_name] = []
                for initialisation_number in range(model_initialisation_strategy):
                    transfer_model, params = initialize_transfer_learner_model(base_model_config, transfer_model_config, dataset_config, modeling_config)
                    transfer_models[transfer_model_name][base_model_name].append((transfer_model, params))
                    initialisation_path = os.path.join(base_model_path, f'init_{initialisation_number + 1}')
                    create_directory(initialisation_path)
                    transfer_model.serialize_or_load(initialisation_path)
                    if os.path.exists(os.path.join(initialisation_path, 'params.pkl')):
                        params = load_pickle_serialization(os.path.join(initialisation_path, 'params.pkl'))
                    else:
                        pickle_serialize(params, os.path.join(initialisation_path, 'params.pkl'))
        return initialize_meta_learner_model_for_multiple_runs
    else:
        return initialize_general_meta_learner_model


def run_transfer_training(config, data, labels, output_path):
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

    retrain_model = modeling_config.get('retrain_model', False)

    data_factory = data_preprocess_factory(factor, mitigation_strategies, factor_path)
    dataset_factory = dataset_initialisation_factory(factor, mitigation_strategies, factor_path)
    run_factory = run_initialisation_factory(factor, mitigation_strategies, factor_path, modeling_config)

    for labelled_train, labelled_validation, test, out_path, data_split_number, label_selection_number in data_factory(data, validation_split, test_split, labels_to_use):
        for train_dataset, test_dataset, dataset_path, data_order_idx in dataset_factory(labelled_train, labelled_validation, test, dataset_config, feature_to_use, out_path, device):
            for transfer_learner, params, transfer_learner_path, run_number, transfer_model_name, base_model_name in run_factory(modeling_config, dataset_path):
                if not transfer_learner.serialization_exists(transfer_learner_path) or retrain_model:
                    print('Training model!')
                    transfer_learner.train(
                        train_dataset,
                        batch_size=params['batch_size'],
                        epochs=params['num_epochs'],
                    )
                    transfer_learner.serialize(transfer_learner_path)
                else:
                    print('Skipping training!')
                    transfer_learner.load_serialization(transfer_learner_path)
                results_path = transfer_learner_path
                if not os.path.exists(os.path.join(results_path, 'results.json')):
                    print(f'Evaluating model!')
                    predictions = transfer_learner.evaluate_in_batch(test_dataset, params['evaluation_batch'])
                    with open(os.path.join(results_path, 'results.json'), 'w') as file:
                        json.dump(
                            {
                                'split_number': data_split_number + 1,
                                'label_selection': label_selection_number + 1,
                                'run_number': run_number + 1,
                                'transfer_model': transfer_model_name,
                                'base_model': base_model_name,
                                'predictions': predictions
                            },
                            file
                        )


def run_parameter_optimisation(config, data, labels, output_path):
    modeling_config = config['modeling'] 
    data['label'] = labels
    test_split = modeling_config['train_test_split']
    validation_split = modeling_config['train_valid_split']
    feature_to_use = modeling_config['feature']
    max_length = modeling_config['dataset_parameters']['max_sentence_length']
    mitigation_strategies = modeling_config['mitigations_strategies']
    device = torch.device(
        'cuda:0'
        if torch.cuda.is_available() and modeling_config['use_gpu']
        else 'cpu'
    )

    parameter_grid = ParameterGrid(modeling_config['parameters_to_search'])

    for data_split_number in range(mitigation_strategies['data_split']):
        print(f'Creating split number {data_split_number + 1}!')
        data_split_path = os.path.join(output_path, 'predictions', str(data_split_number))
        create_directory(data_split_path)

        if not os.path.exists(os.path.join(data_split_path, 'configuration.json')):
            train, validation, test = split_train_validation_test(data, validation_split, test_split)

            labels_to_use = modeling_config['labels_to_use']
            train_labels_to_use = int((1 - validation_split) * labels_to_use)
            validation_labels_to_use = int(validation_split * labels_to_use)
            print(f'Selecting {train_labels_to_use} train labels and {validation_labels_to_use} validation labels!')
            labelled_train = select_labelled_subset(train, train_labels_to_use)

            labelled_validation = select_labelled_subset(validation, validation_labels_to_use)

            serialize_configuration(
                {
                    'train': labelled_train.index.tolist(),
                    'validation': labelled_validation.index.tolist(),
                    'test': test.index.tolist()
                },
                os.path.join(data_split_path, 'configuration.json')
            )

            
        else:
            configuration = load_serialized_configuration(os.path.join(data_split_path, 'configuration.json'))
            labelled_train = data.iloc[configuration['train']]
            labelled_validation = data.iloc[configuration['validation']]
            test = data.iloc[configuration['test']]

        if not os.path.exists(os.path.join(data_split_path, 'grid.pkl')):
            pickle_serialize(parameter_grid, os.path.join(data_split_path, 'grid.pkl'))
        else:
            parameter_grid = load_pickle_serialization(os.path.join(data_split_path, 'grid.pkl'))
        create_directory(os.path.join(data_split_path, 'results'))

        print(f'Using following shapes of train and validation data: \n{labelled_train.shape}\n{labelled_validation.shape}')

        train_X = labelled_train[feature_to_use].tolist()
        train_y = labelled_train['label'].to_numpy()

        validation_X = labelled_validation[feature_to_use].tolist()
        validation_y = labelled_validation['label'].to_numpy()

        test_X = test[feature_to_use].tolist()
        test_y = test['label'].to_numpy()

        train_dataset = MetaTextDataset(
            train_data=(train_X, train_y),
            test_data=(validation_X, validation_y),
            max_length=max_length,
            preembed_data=True,
            device=device,
            transfer_learning=True
        )

        test_dataset = MetaTextDataset(
            train_data=(train_X, train_y),
            test_data=(test_X, test_y),
            max_length=max_length,
            preembed_data=True,
            device=device,
            transfer_learning=True
        )

        num_parameter_runs = 0
        for parameter_run_name in os.listdir(os.path.join(data_split_path, 'results')):
            if parameter_run_name.startswith('run_'):
                parameter_number = int(re.findall(r'\d+', parameter_run_name)[0])
                num_parameter_runs = max(num_parameter_runs, parameter_number + 1)



        for idx, parameter_dictionary in enumerate(parameter_grid):
            if (idx + 1) <= num_parameter_runs:
                continue
            print(f'Using following parameters: {parameter_dictionary}')
            dataset_config = parse_dataset_config(modeling_config['dataset_parameters'], parameter_dictionary)

            for run_number in range(mitigation_strategies['repeated_runs']):
                print(f'Starting run {run_number}!')

                for transfer_model_config in modeling_config['transfer_models']:
                    print(f'Using {transfer_model_config["model"]} transfer learning model!')
                    for base_model_config in modeling_config['base_models']:
                        print(f'Using {base_model_config["model"]} base model!')
                        model_params = base_model_config['params']
                        net = base_model_mapping[base_model_config['model']](
                            dataset_config['max_sentence_length'],
                            dataset_config['embedding_dimension'],
                            model_params['n_filters'],
                            model_params['filter_sizes'],
                            model_params['pool_size'],
                            model_params['hidden_size'],
                            dataset_config['num_classes'],
                            model_params.get('initialise_classifier', True)
                        )
                        loss_function = loss_mapping[model_params['loss_function']]
                        
                        transfer_learner_config = parse_transfer_learner_config(modeling_config['transfer_learning_parameters'], parameter_dictionary)
                        params = parse_transfer_learner_config(transfer_learner_config, transfer_model_config['params'])

                        params['optimizer'] = optimizer_mapping[params['optimizer']] if params.get('optimizer') else None
                        params['loss_functions'] = loss_function

                        transfer_model = transfer_learning_model_mapping[transfer_model_config['model']](
                            model=net,
                            params=params,
                            device=device
                        )
                        transfer_model.train(
                            train_dataset,
                            batch_size=params['batch_size'],
                            epochs=params['num_epochs'],
                        )

                        print('Starting prediction!')
                        predictions = transfer_model.evaluate_in_batch(test_dataset, params['evaluation_batch'])

                        model_name = transfer_model_config['model']
                        configuration_to_save = {
                            'split_number': data_split_number,
                            'labels_used': 1000,
                            'run_number': run_number,
                            'transfer_model': transfer_model_config['model'],
                            'base_model': base_model_config['model'],
                            'predictions': predictions
                        }
                        for key, value in parameter_dictionary.items():
                            configuration_to_save[key] = value
                        file_name = f'run_{idx}.json'
                        with open(os.path.join(data_split_path, 'results', file_name), 'w') as file:
                            json.dump(configuration_to_save, file)
                        save_results_json(configuration_to_save, data_split_path)
