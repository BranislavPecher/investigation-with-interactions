from __future__ import annotations
import pandas as pd

from sklearn.base import TransformerMixin
from typing import List, Optional, Iterable
from src.preprocessing.decorators import transformer_time_measurement_decorator

class TextConcatenate(TransformerMixin):
    """Concatenates selected columns together using specified separator.

    Args:
        output_column (str): Column where the concatenated text will be saved. 
        columns (Optional[List[str]]): Columns to be concatenated together. Defaults to None.
        separator: (str): Separator used to concatenate texts together. Defaults to empty space.
    """

    def __init__(self, output_column: str, columns: Optional[List[str]] = None, separator: Optional[str] = ' ') -> None:
        self.columns = columns
        self.output_column = output_column
        self.separator = separator

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> TextConcatenate:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            LowercaseTransformer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('TextConcatenate')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concatenates columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df[self.output_column] = df[self.columns].agg(self.separator.join, axis=1)

        return df