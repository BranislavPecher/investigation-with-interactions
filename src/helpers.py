import os
import pandas as pd
import json
import pickle

from datetime import datetime
from typing import List, Union, Any, Tuple


def create_directory(path: str) -> None:
    """Create directory at given path.

    Args:
        path (str): Path to create directory.
    """
    if not os.path.exists(path):
        print(f'>> Creating directory: "{path}".')
        os.makedirs(path)


def get_timestamp_string() -> str:
    """Get current datetime timestamp string.

    Returns:
        str: Current datetime timestamp string.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S")


def init_pipeline_folders(path: str) -> None:
    """Initialize folders for pipeline run.

    Args:
        path (str): Root path for pipeline run results.
    """
    create_directory(path)
    create_directory(f'{path}/preprocessing')
    create_directory(f'{path}/models')
    create_directory(f'{path}/evaluation')
    create_directory(f'{path}/data')
    create_directory(f'{path}/scores')
    create_directory(f'{path}/predictions')


def load_json_config(path: str) -> dict:
    """Load pipeline config from json file.

    Args:
        path (str): Path to json file to load.

    Returns:
        dict: Loaded config as dictionary.
    """
    print(f'>> Loading config "{path}".')
    with open(path, 'r') as f:
        return json.load(f)


def create_config_copy(config: dict, path: str) -> None:
    """Create copy of config as json file.

    Args:
        config (dict): Config in form of dictionary.
        path (str): Path where config will be stored.
    """
    with open(path, 'w') as f:
        json.dump(config, f)


def load_records(entity: str, data_dir: str, ids: List[int] = None) -> List[dict]:
    """Load saved json files records.

    Args:
        entity (str): Type of entity records to be loaded.
        data_dir (str): Directory where data are stored.
        ids (List[int], optional): List of ids to filter records. Defaults to None.

    Returns:
        List[dict]: List of records (dictionary objects).
    """
    records = []
    folder_path = f'{data_dir}/{entity.replace("-", "_")}'
    for file_name in os.listdir(folder_path):
        if ids is not None:
            if int(file_name.split('.')[0]) not in ids:
                continue
        with open(f'{folder_path}/{file_name}', 'r') as f:
            records.append(json.load(f))

    return records


def load_data(path: Union[str, List[str]], sample: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data from .csv file (or files) to dataframe.

    In case that more file paths are provided, all data will be loaded and
    concatenated with ignoring index.

    Args:
        path (Union[str, List[str]]): Path (or list of paths) to .csv file to be loaded.
        sample (int): Number of samples to return. If None, return all. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Dataframe with loaded data, divided to features and labels.
    """
    if isinstance(path, str):
        return pd.read_csv(path)

    dataframes = []
    for file_path in path:
        print(f'>> Loading data "{file_path}.')
        dataframes.append(pd.read_csv(file_path))
    df = pd.concat(dataframes, ignore_index=True)

    if sample is not None:
        df = df.sample(sample)

    if 'veracity' in df:
        # y = df.loc[:, 'veracity'].copy(deep=True)
        y = df['veracity']
        df.drop(columns=['veracity'], inplace=True)
    elif 'label' in df:
        # y = df.loc[:, 'label'].copy(deep=True)
        y = df['label']
        df.drop(columns=['label'], inplace=True)
    else:
        y = None

    return df, y


def join_strings_list(strings_list_data: Union[List[str], str, None]) -> Union[None, str]:
    """Join strings list into one string.

    Function handles more cases of potential variable values.

    Args:
        strings_list_data (Union[List[str], str, None]): Strings list data to be joined.

    Returns:
        Union[None, str]: Joined string or None, if input was also None.
    """
    if strings_list_data is None:
        return None
    if isinstance(strings_list_data, str):
        return strings_list_data
    return ', '.join([value for value in strings_list_data if value is not None])


def serialize_object(obj: Any, path: str) -> None:
    """Serialie (pickle) object.

    Args:
        obj (Any): Object to be serialized.
        path (str): Path to file where object will be serialized.
    """
    print(f'>> Serializing object to {path}.')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_serialized_object(path: str) -> Any:
    """Load serialized (pickled) object.

    Args:
        path (str): Path to serialized object to be loaded.

    Returns:
        Any: Loaded serialized object.
    """
    print(f'>> Loading serialized object from {path}.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_serialized_data(folder: str, data_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load serialized pre-processed train/test data.

    Args:
        folder (str): Folder where pre-processed data are stored.
        data_type (str): Type of data (one of 'train', 'test').

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Pre-processed train/test data.
    """
    print(f'>> Loading pre-processed {data_type} data from {folder}.')
    x = load_serialized_object(f'{folder}/x_{data_type}.pkl')
    y = load_serialized_object(f'{folder}/y_{data_type}.pkl')

    return x, y


def serialize_data(x: pd.DataFrame, y: pd.Series, folder: str, data_type: str) -> None:
    """Serialize pre-processed train/test data.

    Be careful when updating, another helper (load_serialized_preprocessed_data) is using
    pre-defined names of files.

    Args:
        x (pd.DataFrame): Features dataframe.
        y (pd.Series): Labels series.
        folder (str): Folder where data will be serialized.
        data_type (str): Type of data (one of 'train', 'test').
    """
    print(f'>> Serializing pre-processed {data_type} data to {folder}.')
    serialize_object(x, path=f'{folder}/x_{data_type}.pkl')
    serialize_object(y, path=f'{folder}/y_{data_type}.pkl')


def log_text(text: str, path: str) -> None:
    """Log text into .txt file.

    Args:
        text (str): Text to be logged.
        path (str): Path where text should be logged.
    """
    with open(path, 'a') as f:
        f.write(text)
        f.write('\n')


def print_dict_pretty(dictionary: dict, indent: str = '  ') -> None:
    """Print dictionary in pretty way.

    Args:
        dictionary (dict): Dictionary to be printed.
        indent (str, optional): Prefix indent before each key: value print-out. Defaults to '  '.
    """
    for key, value in dictionary.items():
        print(f'{indent}{key}: {value}')
