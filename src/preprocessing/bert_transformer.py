from sklearn.base import TransformerMixin
from typing import List, Iterable, Set, Optional
from transformers import BertTokenizer, BertModel
from src.preprocessing.decorators import transformer_time_measurement_decorator
import torch
import pandas as pd
import re

class BertTransformer(TransformerMixin):
    """Transforms texts in selected columns to BERT vectors using HuggingFace transformers
    Args:
        columns (Optional[List[str]]): Columns to be transformed to vectors. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None, size: Optional[int] = 300):
        self.columns = columns
        self.size = size

        self.embedding_model = None
        self.tokenizer = None

    def fit(self, df: pd.DataFrame, y: Iterable = None):
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            BertTransformer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_model = BertModel.from_pretrained('bert-base-uncased')

        return self

    @transformer_time_measurement_decorator('BertTransformer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            embeddings = self.calculate_embeddings_from_texts_(df[column])
            df[f'{column}_embedded'] = embeddings
            # df.drop([column], axis=1, inplace=True)
        return df

    def calculate_embeddings_from_texts_(self, texts: str):
        sentences = []
        for sentence in texts:
            tokens = self.tokenizer.tokenize(f'[CLS] {sentence} [SEP]')
            tokens = [token for token in tokens if not re.search(r'^\W+$', token)]
            tokens = tokens[:self.size + 2]
            if tokens[-1] != '[SEP]':
                tokens[-1] = '[SEP]'
            sentences.append(tokens)
        #sentences = [self.tokenizer.tokenize(f'[CLS] {sentence} [SEP]') for sentence in texts]
        sentences_tokens = [self.tokenizer.convert_tokens_to_ids(sentence) for sentence in sentences]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model.to(device)
        embeddings = []
        self.embedding_model.eval()
        with torch.no_grad():
            for idx, sentence in enumerate(sentences_tokens):
                tensor = torch.tensor(sentence, dtype=torch.long).reshape(1, -1).to(device)
                embedding = self.embedding_model(tensor)[0].cpu().detach().numpy()
                embeddings.append(embedding[:, 1:-1])
                del tensor
        return embeddings
