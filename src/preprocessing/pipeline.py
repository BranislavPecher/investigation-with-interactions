from typing import Union, List
from sklearn.pipeline import Pipeline
from src.preprocessing.transformers_base import ColumnsSelector, DuplicatesFilter, NaNsFilter
from src.preprocessing.transformers_text import TextCleaner, TextLengthFilter, LanguageFilter, \
    StopwordsFilter, LowercaseTransformer, Lemmatizer
from src.preprocessing.vectorizer import Vectorizer
from src.preprocessing.bert_transformer import BertTransformer
from src.preprocessing.text_concatenate import TextConcatenate
from src.preprocessing.transformers_features import SentimentAnalyzer, ReadabilityScoreAnalyzer, \
    UppercaseRatioCounter, URLsCounter, TokensCounter, Scaler, DimensionalityReducer, \
    EmbeddingVectorsSimilarityCalculator, CharacterCounter


transformer_classes = [
    # General transformers
    ColumnsSelector, DuplicatesFilter, NaNsFilter,

    # Text transformers
    TextCleaner, TextLengthFilter, LanguageFilter, StopwordsFilter,
    LowercaseTransformer, Vectorizer, Lemmatizer, BertTransformer, TextConcatenate,

    # Feature engineering
    SentimentAnalyzer, ReadabilityScoreAnalyzer, UppercaseRatioCounter, URLsCounter, TokensCounter,
    CharacterCounter, Scaler, DimensionalityReducer, EmbeddingVectorsSimilarityCalculator
]
transformer_name_mapping = {
    transformer_class.__name__: transformer_class for transformer_class in transformer_classes
}


def get_preprocessing_pipeline(config: List[dict], serialize_folder: str = None) -> Pipeline:
    """Get pre-processing pipeline according to config.

    Args:
        config (List[dict]): Config with pipeline structure.
        serialize_folder (str, optional): Folder where models can be serialized. Defaults to None.

    Returns:
        Pipeline: Constructed pipeline.

    Raises:
        ValueError: In case unsupported transformer is provided in config.
    """
    def transform_arguments(arguments: Union[dict, None]) -> dict:
        arguments = arguments if arguments is not None else {}
        if arguments.get('serialize'):
            arguments.pop('serialize')
            arguments['serialize_folder'] = serialize_folder
        return arguments

    try:
        return Pipeline([
            (
                transformer_config['transformer'],
                transformer_name_mapping[transformer_config['transformer']](
                    **transform_arguments(transformer_config['params'])
                )
            )
            for transformer_config in config
        ])
    except KeyError as key_error_name:
        raise ValueError(f'Sorry, but {key_error_name} transformer is not supported. Please, use '
                        f'one of the following: {", ".join(transformer_name_mapping.keys())}.')
