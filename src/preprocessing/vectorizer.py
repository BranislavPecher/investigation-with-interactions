from __future__ import annotations
import pandas as pd
import numpy as np

from typing import List, Iterable, Set, Optional
from sklearn.base import TransformerMixin
from src.preprocessing.decorators import transformer_time_measurement_decorator
from src.embedding_model import EmbeddingModel
from src.helpers import load_serialized_object
# from gensim.models.fasttext import load_facebook_model


class Vectorizer(TransformerMixin):
    """Transform texts in selected columns to vectors.

    Args:
        columns (Optional[List[str]]): Columns to be transformed to vectors. Defaults to None.
        model_name (str, optional): Model name to be used (according to files conventions).
            Defaults to fastText.
        path_prefix (str, optional): Path prefix to model serialized files. Defaults to
            'serialized/embedding_models'.
        fine_tune (bool, optional): Whether embedding model should be fine-tuned. This only works
            for fastText models. Be careful to provide path to embedding model that can be
            fine-tuned (not only keyed vectors). Also, 'serialize_folder' must be provided in
            case of fine-tuning. If you want to use also additional data for fine-tuning,
            simply provide also 'additional_data_path' parameter to csv file - but be careful,
            data must be already pre-processed in that csv file. Defaults to False.
        serialize_folder (str, optional): Folder where fine-tuned embedding model will be stored
            in case 'fine_tune' parameter is True. Defaults to None.
        additional_data_path (str, optional): Path to additional data (csv file) in case of
            fine-tuning ('fine_tune' parameter equal to True). Be careful, data must be already
            pre-processed. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None, model_name: str = 'fastText',
                 path_prefix: str = 'serialized/embedding_models', fine_tune: bool = False,
                 serialize_folder: str = None, additional_data_path: str = None) -> None:
        self.columns = columns
        self.model_name = model_name
        self.fine_tune = fine_tune
        self.serialize_folder = serialize_folder
        self.additional_data_path = additional_data_path
        self.path_prefix = path_prefix

        self.embedding_model = None

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> Vectorizer:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            Vectorizer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)

        model_name = self.model_name
        model_path_prefix = self.path_prefix

        if self.fine_tune and self.serialize_folder is not None:
            df_ft = df.copy()[self.columns]
            if self.additional_data_path is not None:
                df_additional = load_serialized_object(path=self.additional_data_path)
                df_ft = pd.concat([df_ft, df_additional[self.columns]], ignore_index=True)

            df_ft['_'.join(self.columns)] = df_ft.apply(
                lambda row: ' '.join([row[column] for column in self.columns]), axis=1)

            new_sentences = df_ft['_'.join(self.columns)].values
            # model = load_facebook_model(f'{self.path_prefix}/{self.model_name}.bin')
            model = {}

            model.build_vocab(new_sentences, update=True)
            model.train(sentences=new_sentences, epochs=model.epochs,
                        total_examples=len(new_sentences))

            model_name = f'{self.model_name}_tuned'
            model_path_prefix = self.serialize_folder
            model.wv.save_word2vec_format(f'{model_path_prefix}/{model_name}.bin', binary=True)

        self.embedding_model = EmbeddingModel(model_name=model_name,
                                              path_prefix=model_path_prefix)

        return self

    @transformer_time_measurement_decorator('Vectorizer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            embeddings = self.embedding_model.get_embeddings_from_texts(df[column])
            df_embeddings = pd.DataFrame(
                {f'{column}_{idx}': embedding
                 for idx, embedding in enumerate(np.array(embeddings).T)},
                index=df.index
            )
            df = df.join(df_embeddings)
            df.drop([column], axis=1, inplace=True)

        return df
