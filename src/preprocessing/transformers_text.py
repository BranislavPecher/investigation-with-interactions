from __future__ import annotations
import pandas as pd
import re

from typing import List, Iterable, Set, Optional
from langdetect import detect
from sklearn.base import TransformerMixin
from src.preprocessing.decorators import transformer_time_measurement_decorator
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, WordNetLemmatizer


class TextCleaner(TransformerMixin):
    """Clean text attribute.

    During text cleaning, following operations are performed:
    - removing html tags,
    - removing URLs,
    - removing all special characters except letters.

    Args:
        columns (Optional[List[str]]): Text columns that should be cleaned. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> TextCleaner:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            TextCleaner: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('TextCleaner')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[column].fillna('', inplace=True)
            df[column] = df[column].astype(str)

            # Remove HTML tags
            df[column] = df[column].apply(lambda text: re.sub(r'<[/a-zA-Z]+>', ' ', text))

            # Remove URLs
            df[column] = df[column].apply(
                lambda text: re.sub(r'(www|http:|https:)+[^\s]+[\w]', ' ', text)
            )

            # Remove all special characters and strip text
            df[column] = df[column].apply(lambda text: re.sub(r'[^a-zA-Z]+', ' ', text).strip())

        return df


class TextLengthFilter(TransformerMixin):
    """Filter rows with too short text in specific columns.

    Args:
        columns (Optional[List[str]]): Columns to perform length filter by. Defaults to None.
        min_length (int): Length lower boundary threshold as number of words. Defaults to 200.
        max_length (int): Length upper boundary threshold as number of words. Defaults to 200.
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 min_length: int = 200, max_length: int = 600) -> None:
        self.columns = columns
        self.min_length = min_length
        self.max_length = max_length

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> TextLengthFilter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            TextLengthFilter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self


    @transformer_time_measurement_decorator('TextLengthFilter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[f'{column}_words_count'] = df[column].apply(lambda text: len(text.split()))
            df = df[(df[f'{column}_words_count'] >= self.min_length) &
                    (df[f'{column}_words_count'] <= self.max_length)]
            df.drop([f'{column}_words_count'], axis=1, inplace=True)

        return df


class LanguageFilter(TransformerMixin):
    """Filter rows with different language.

    Args:
        columns (Optional[List[str]]): Columns to filter by language in. Defaults to None.
        language (str): Language that is the only one correct. Defaults to en.
    """

    def __init__(self, columns: Optional[List[str]] = None, language: str = 'en') -> None:
        self.columns = columns
        self.language = language

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> LanguageFilter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            LanguageFilter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('LanguageFilter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            lang_col_name = f'{column}_lang'
            df[lang_col_name] = df[column].apply(lambda text: detect(text[:min(300, len(text))]))
            df = df[df[lang_col_name] == self.language]

            df.drop([lang_col_name], axis=1, inplace=True)

        return df


class LowercaseTransformer(TransformerMixin):
    """Transform texts in selected columns to lower-case.

    Args:
        columns (Optional[List[str]]): Columns to be lower-cased. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> LowercaseTransformer:
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

    @transformer_time_measurement_decorator('LowercaseTransformer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[column] = df[column].apply(lambda text: text.lower())

        return df


class StopwordsFilter(TransformerMixin):
    """Filter stop-words from texts in selected columns.

    Args:
        columns (Optional[List[str]]): Columns to filter stopwords in. Defaults to None.
        stop_words (Optional[Set[str]]): Stopwords to filter, if None, english
            stopwords from nltk library will be used. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None,
                 stop_words: Optional[Set[str]] = None) -> None:
        self.columns = columns
        self.stopwords = stop_words if stop_words is not None else set(stopwords.words('english'))

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> StopwordsFilter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            StopwordsFilter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('StopwordsFilter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[column] = df[column].apply(lambda text:
                ' '.join([token for token in text.split() if token not in self.stopwords]))

        return df


class Lemmatizer(TransformerMixin):
    """Lemmatize words in text columns.

    Make sure that words are lower-cased and special characters are removed before
    using this transformer!

    Args:
        columns (Optional[List[str]]): Columns to lemmatize words in. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> Lemmatizer:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            Lemmatizer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('Lemmatizer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[column] = df[column].apply(self.lemmatize)

        return df

    def lemmatize(self, text: str) -> str:
        """Lemmatize words of given text.

        Args:
            text (str): Text of which words will be lemmatized.

        Returns:
            str: Text with words being lemmatized.
        """
        mapping = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}

        lemmatized_words = [
            self.lemmatizer.lemmatize(
                token, mapping.get(pos_tag([token])[0][1][0].upper(), wordnet.NOUN))
            for token in text.split()
        ]

        return ' '.join(lemmatized_words)


