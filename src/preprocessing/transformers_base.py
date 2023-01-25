from __future__ import annotations
import pandas as pd

from typing import List, Union, Iterable, Optional
from src.preprocessing.decorators import transformer_time_measurement_decorator
from sklearn.base import TransformerMixin


class ColumnsSelector(TransformerMixin):
    """Select only some columns from dataframe.

    Args:
        columns (Union[List[str], str]): Columns to be selected.
        drop (bool): Whether passed columns should be dropped. Defaults to False.
    """

    def __init__(self, columns: Union[List[str], str], drop: bool = False) -> None:
        self.columns = [columns] if isinstance(columns, str) else columns
        self.drop = drop

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> ColumnsSelector:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            ColumnsSelector: Self object.
        """
        return self

    @transformer_time_measurement_decorator('ColumnsSelector')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        return df[self.columns] if not self.drop else df[set(df.columns) - set(self.columns)]


class NaNsFilter(TransformerMixin):
    """Filter NaN values by column (or all columns).

    Args:
        columns (Optional[List[str]]): Columns to remove NaN values in. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> NaNsFilter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            NaNsFilter: Self object.
        """
        return self

    @transformer_time_measurement_decorator('NaNsFilter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df.dropna(subset=self.columns, inplace=True)
        return df


class DuplicatesFilter(TransformerMixin):
    """Filter duplicates by column (or all columns).

    Args:
        columns (Optional[List[str]]): Columns to remove duplicates by. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> DuplicatesFilter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            DuplicatesFilter: Self object.
        """
        return self

    @transformer_time_measurement_decorator('DuplicatesFilter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df.drop_duplicates(subset=self.columns, inplace=True)
        return df
