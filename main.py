import os
import argparse
import warnings

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.helpers import load_json_config, load_data, init_pipeline_folders, serialize_object, \
    load_serialized_data, serialize_data, load_serialized_object, get_timestamp_string, \
    create_config_copy
from src.modeling.meta_training import run_meta_training, run_parameter_optimisation
from src.modeling.transfer_training import run_transfer_training, run_parameter_optimisation as run_transfer_parameter_optimisation
from src.preprocessing.pipeline import get_preprocessing_pipeline
from typing import Union, List
import datasets
import pandas as pd


def parse_arguments() -> Union[List[str], str]:
    """Parse script call arguments.

    Returns:
        Union[List[str], str]: Paths to config files in .json format.
    """
    parser = argparse.ArgumentParser(prog='Training pipeline',
                                     description='Main pipeline for training models.')
    parser.add_argument('-c', '--config', nargs='+', required=True,
                        help='Path or paths to config files in .json format.')
    args = parser.parse_args()

    return args.config


def run_pipeline(config_path: str) -> None:
    """Run pipeline with config from file provided as argument.

    Args:
        config_path (str): Path to config to run pipeline with.
    """
    config = load_json_config(path=config_path)

    # Create folder for results
    if config.get('use_timestamp_dir', True):
        output_folder = f'{config["output_folder"]}/{get_timestamp_string()}'
    else:
        output_folder = f'{config["output_folder"]}'
    init_pipeline_folders(output_folder)
    create_config_copy(config=config, path=f'{output_folder}/config.json')

    print(f'>> Running pipeline for config "{config_path}".\n'
          f'>> Output folder: "{output_folder}".')

    x_unlabeled = None
    use_test_data = False

    if config['preprocessing'].get('serialized_folder') is not None:
        # Load data that have been already pre-processed in another pipeline run
        serialized_folder = config['preprocessing']['serialized_folder']
        use_test_data = config['preprocessing'].get('load_test_data', False)
        x_train, y_train = load_serialized_data(folder=serialized_folder, data_type='train')
        if use_test_data:
            x_test, y_test = load_serialized_data(folder=serialized_folder, data_type='test')

        unlabeled_data_path = f'{serialized_folder}/x_unlabeled.pkl'
        if os.path.exists(unlabeled_data_path):
            x_unlabeled = load_serialized_object(path=unlabeled_data_path)
    else:
        # 1. Loading the data
        include_unlabeled = config['data'].get('unlabeled') is not None
        if config['data']['source'] == 'local':
            df_train, y_train = load_data(config['data']['labeled'])

            if include_unlabeled:
                df_unlabeled, _ = load_data(config['data']['unlabeled']['paths'],
                                            sample=config['data']['unlabeled'].get('sample'))
        elif config['data']['source'] == 'datasets':
            path = config['data']['path']
            split = config['data'].get('split')
            dataset = datasets.load_dataset(path, split) if split is not None else datasets.load_dataset(path)
            full_data = pd.concat([pd.DataFrame(dataset[key]) for key in config['data'].get('keys', [])], ignore_index=True)
            df_train = full_data[config['data'].get('data_fields', ['text'])]
            y_train = full_data[config['data'].get('label_field', 'label')]

        else:
            raise NotImplementedError('Selected data source not implemented yet! Choose one from (local | datasets).')

        # 2. Pre-processing and feature engineering
        print('>> Pre-processing of data.')
        pipeline = get_preprocessing_pipeline(config['preprocessing']['transformers'],
                                              serialize_folder=output_folder)

        print('>> Fitting pipeline and pre-processing train data.')
        x_train = pipeline.fit_transform(df_train, y_train)
        y_train = y_train.loc[x_train.index]

        if include_unlabeled:
            print('>> Pre-processing unlabeled data.')
            x_unlabeled = pipeline.transform(df_unlabeled)

        if config['preprocessing']['serialize_data']:
            print(x_train.shape)
            print(y_train.shape)
            serialize_data(x=x_train, y=y_train, folder=f'{output_folder}/data', data_type='train')
            if split_ratio is not None:
                serialize_data(x=x_test, y=y_test, folder=f'{output_folder}/data', data_type='test')
            if include_unlabeled:
                serialize_object(x_unlabeled, path=f'{output_folder}/data/x_unlabeled.pkl')

    # Transform labels to int
    y_train = y_train.apply(lambda x: 1 if x in [True, 'reliable', 1] else 0)
    y_test = y_test.apply(lambda x: 1 if x in [True, 'reliable', 1] else 0) if use_test_data else None

    # 3. Modeling
    if config['modeling'].get('meta_learning', False):
        if config['modeling'].get('parameter_optimisation', False):
            print('Running basic meta parameter optimisation')
            run_parameter_optimisation(config, x_train, y_train, output_folder)
        else:
            print('Running meta-learning experiment!')
            run_meta_training(config, x_train, y_train, output_folder)
    elif config['modeling'].get('transfer_learning', False):
        if config['modeling'].get('parameter_optimisation', False):
            print('Running basic transfer learning parameter optimisation')
            run_transfer_parameter_optimisation(config, x_train, y_train, output_folder)
        else:
            print('Running transfer-learning experiment!')
            run_transfer_training(config, x_train, y_train, output_folder)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # All warnings checked before

    config_files = parse_arguments()
    config_files = [config_files] if isinstance(config_files, str) else config_files

    for config_file in config_files:
        run_pipeline(config_path=config_file)
