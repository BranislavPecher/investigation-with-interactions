from __future__ import annotations
import pandas as pd
import re
import pyphen
import math

from typing import List, Iterable, Optional, Union, Tuple
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.decorators import transformer_time_measurement_decorator
from src.helpers import serialize_object
#from umap import UMAP
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class SentimentAnalyzer(TransformerMixin):
    """Create sentiment feature from texts in selected columns.

    For calculating sentiment, we are using VADER technique:
    Hutto, C.J. & Gilbert, Eric. (2015). VADER: A Parsimonious Rule-based Model for Sentiment
    Analysis of Social Media Text.
    Proceedings of the 8th International Conference on Weblogs and Social Media, ICWSM 2014.

    Args:
        columns (Optional[List[str]]): Columns to calculate sentimnet of. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> SentimentAnalyzer:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            SentimentAnalyzer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('SentimentAnalyzer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[f'{column}_sentiment'] = df[column].apply(self.calculate_vader_sentiment)

        return df

    @staticmethod
    def calculate_vader_sentiment(text: str) -> float:
        """Calculate VADER sentiment score.

        More about VADER:
            Hutto, C.J. & Gilbert, Eric. (2015).
            VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
            Proceedings of the 8th International Conference on Weblogs and Social Media, ICWSM 2014. 

        Args:
            text (str): Text to calculate VADER sentiment score of.

        Returns:
            float: VADER sentiment score.
        """
        vader_analyzer = SentimentIntensityAnalyzer()
        return vader_analyzer.polarity_scores(text)['compound']

class ReadabilityScoreAnalyzer(TransformerMixin):
    """Create readability score features from texts in selected columns.

    Multiple readability indexes are calculated:
        - Flesch Reading Ease
        - Flesch-Kincaid Grade Level
        - Coleman-Liau Index
        - Automated Readability Index (ARI)

    All readability formulas were taken from: https://readabilityformulas.com

    Args:
        columns (Optional[List[str]]): Columns to calculate readability score of. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> ReadabilityScoreAnalyzer:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            ReadabilityScoreAnalyzer: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('ReadabilityScoreAnalyzer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        suffix_mapping = {
            '_ri_fre': self.calculate_automated_readability_index,
            '_ri_fk': self.calculate_flesch_kincaid_readability_score,
            '_ri_cli': self.calculate_coleman_liau_index,
            '_ri_ari': self.calculate_automated_readability_index
        }

        for column in self.columns:
            for suffix, calculate_index_function in suffix_mapping.items():
                df[f'{column}{suffix}'] = df[column].apply(calculate_index_function)

        return df

    @staticmethod
    def calculate_flesch_readability_score(text: str) -> float:
        """Calculate Flesch readability easy score for given text.

        Flesch reading-ease score is calculated as follows:
            206.835 - 1.015 (total_words / total_sentences) - 84.6 (total_syllables / total_words)

        Higher score means that text is easier to read, texts with lower score are harder to read.
        Examples:
            100.00 – 90.00	Very easy to read. Easily understood by an average 11-year-old student.
            ...
            30.0–10.0	Very difficult to read. Best understood by university graduates.
        Source: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

        Args:
            text (str): Text to calculate readability score of.

        Returns:
            float: Flesch reading-ease score of given text.
        """
        dic = pyphen.Pyphen(lang='en')
        text_prep = re.sub(r'[^a-zA-Z]', ' ', text)
        n_syllables = sum([len(dic.inserted(word).split('-')) for word in text_prep.split()])
        n_words = max(len(text_prep.split()), 1)
        n_sentences = max(len(sent_tokenize(text)), 1)

        return 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)

    @staticmethod
    def calculate_flesch_kincaid_readability_score(text: str) -> float:
        """Calculate Flesch-Kincaid redatability score for given text.

        Flesch-Kincaid grade level readability score is calculated as follows:
            FKRA = (0.39 x ASL) + (11.8 x ASW) - 15.59,
            where ASL is average sentence length (# words / # sentences) and ASW is average
            number of syllable per word (# syllables / # words)

        Args:
            text (str): Text to calculate readability score of.

        Returns:
            float: Flesch-Kincaid readability score of given text.
        """
        dic = pyphen.Pyphen(lang='en')
        text_prep = re.sub(r'[^a-zA-Z]', ' ', text)
        n_syllables = sum([len(dic.inserted(word).split('-')) for word in text_prep.split()])
        n_words = max(len(text_prep.split()), 1)
        n_sentences = max(len(sent_tokenize(text)), 1)

        return 0.39 * n_words / n_sentences + 11.8 * n_syllables / n_words - 15.59

    @staticmethod
    def calculate_coleman_liau_index(text: str) -> float:
        """Calculate Coleman-Liau index for given text.

        Coleman-Liau readability index is calculated as follows:
            CLI = 0.0588 x L - 0.296 x S - 15.8,
            where L is the average number of letters per 100 words and
            S is the average number of sentences per 100 words.

        Args:
            text (str): Text to calculate readability index of.

        Returns:
            float: Coleman-Liau readability index.
        """
        word_tokens = word_tokenize(text)
        word_tokens = [word for word in word_tokens if word.isalnum()]
        n_words = max(len(word_tokens), 1)
        n_letters = max(len(''.join(word_tokens)), 1)
        n_sentences = max(len(sent_tokenize(text)), 1)

        l = n_letters / n_words * 100
        s = n_sentences / n_words * 100

        return 0.0588 * l - 0.296 * s - 15.8

    @staticmethod
    def calculate_automated_readability_index(text: str) -> int:
        """Calculate Automated Readability Index for given text.

        Automated Readability Index is calculated as follows:
            ARI = 4.71 * (#characters / #words) + 0.5 * (#words / #sentences) - 21.43

        Args:
            text (str): Text to calculate readability index of.

        Returns:
            int: Automated readability index.
        """
        word_tokens = word_tokenize(text)
        word_tokens = [word for word in word_tokens if word.isalnum()]
        n_words = max(len(word_tokens), 1)
        n_characters = max(len(''.join(word_tokens)), 1)
        n_sentences = max(len(sent_tokenize(text)), 1)

        return math.ceil(4.71 * (n_characters / n_words) + 0.5 * (n_words / n_sentences) - 21.43)


class UppercaseRatioCounter(TransformerMixin):
    """Create uppercase ratio feature from texts in selected columns.

    Args:
        columns (Optional[List[str]]): Columns to count uppercase ratio of. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> UppercaseRatioCounter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            UppercaseRatioCounter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('UppercaseRatioCounter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[f'{column}_uppercase_ratio'] = df[column].apply(self.calculate_uppercase_ratio)

        return df

    def calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate uppercase ratio from given text.

        Args:
            text (str): Text to calculate uppercase characters ratio.

        Returns:
            float: Ratio of uppercase characters used.
        """
        return len([character for character in text if character.isupper()]) / len(text)


class TokensCounter(TransformerMixin):
    """Count tokens (characters, words, sentences) in text columns.

    Args:
        columns (Optional[List[str]]): Columns to count tokens in. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> TokensCounter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            TokensCounter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('TokensCounter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[f'{column}_n_characters'] = df[column].apply(lambda text: len(str(text)))
            df[f'{column}_n_words'] = df[column].apply(
                lambda text: len([word for word in word_tokenize(text) if word.isalnum()]))
            df[f'{column}_n_sentences'] = df[column].apply(lambda text: len(sent_tokenize(text)))

        return df


class URLsCounter(TransformerMixin):
    """Count urls in text columns.

    Args:
        columns (Optional[List[str]]): Columns to count urls in. Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> URLsCounter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            URLsCounter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('URLsCounter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for column in self.columns:
            df[f'{column}_urls_count'] = df[column].apply(
                lambda text: len(re.findall(r'(www|http:|https:)+[^\s]+[\w]', text)))

        return df


class CharacterCounter(TransformerMixin):
    """Count given characters in text columns.

    Args:
        characters (List[str]): List of characters to count.
        columns (Optional[List[str]]): Columns to characters in. Defaults to None.
    """

    def __init__(self, characters: List[str], columns: Optional[List[str]] = None) -> None:
        self.characters = characters
        self.columns = columns

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> CharacterCounter:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            CharacterCounter: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)
        return self

    @transformer_time_measurement_decorator('CharacterCounter')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        for character in self.characters:
            for column in self.columns:
                df[f'{column}_{character}_count'] = df[column].apply(
                    lambda text: str(text).count(character))

        return df


class Scaler(TransformerMixin):
    """Scale values in column (or all columns).

    Args:
        columns (Optional[List[str]]): Columns to scale data in. Defaults to None.
        serialize_folder (str, optional): Folder where scaler model will be serialized.
            Defaults to None.
    """

    def __init__(self, columns: Optional[List[str]] = None, serialize_folder: str = None) -> None:
        self.columns = columns
        self.scaler = StandardScaler()
        self.serialize_folder = serialize_folder

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> Scaler:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            Scaler: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)

        self.scaler.fit(df[self.columns])
        if self.serialize_folder is not None:
            serialize_object(self.scaler, path=f'{self.serialize_folder}/preprocessing/scaler.pkl')
        return self

    @transformer_time_measurement_decorator('Scaler')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df[self.columns] = self.scaler.transform(df[self.columns])
        return df


class DimensionalityReducer(TransformerMixin):
    """Transformer for dimensionality reduction of embedding features.

    Note: Provided column names are used as prefixes of real embedding features, e.g. 'title'
    will be used as prefix to features 'title_0' - 'title_n', where n is embeddings_size.

    Args:
        columns (List[str]): Column prefixes to reduce dimensionality in.
        method (str, optional): Method to be used for dimensionality reduction. Can be either
            'pca' or 'umap'. Defaults to 'pca'.
        embeddings_size (int, optional): Size of embedding vectors. Defaults to 300.
        n_components (Optional[Union[int, float]]): Number of components to preserve.
            Defaults to 50.
        serialize_folder (str, optional): Folder where dimensionality reduction model will
            be serialized. Defaults to None.
    """

    def __init__(self, columns: List[str], method: str = 'pca', embeddings_size: int = 300,
                 n_components: Optional[Union[int, float]] = 50,
                 serialize_folder: str = None) -> None:
        self.columns = columns
        self.column_names = [f'{column}_{x}' for column in columns for x in range(embeddings_size)]

        self.method = method
        self.embeddings_size = embeddings_size
        self.n_components = n_components
        self.model = PCA(n_components=n_components) if method == 'pca' \
            else UMAP(n_components=n_components)
        self.serialize_folder = serialize_folder

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> DimensionalityReducer:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            DimensionalityReducer: Self object.
        """
        self.model.fit(df[self.column_names])
        if self.serialize_folder is not None:
            serialize_object(self.model,
                             path=f'{self.serialize_folder}/preprocessing/{self.method}.pkl')
        return self

    @transformer_time_measurement_decorator('DimensionalityReducer')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        new_features_names = [f'{self.method}_{x}' for x in range(self.n_components)]
        df_reduced = pd.DataFrame(self.model.transform(df[self.column_names]), index=df.index,
                                  columns=new_features_names)

        return pd.concat([df.drop(self.column_names, axis=1), df_reduced], axis=1, join='inner')


class EmbeddingVectorsSimilarityCalculator(TransformerMixin):
    """Calculate cosine similarity between two text features transformed to embedding vectors.

    Note: Provided column names are used as prefixes of real embedding features, e.g. 'title'
    will be used as prefix to features 'title_0' - 'title_n', where n is embeddings_size.

    Args:
        columns (Tuple[str, str]): Column prefixes to calculate similarity of.
        embeddings_size (int, optional): Size of embedding vectors. Defaults to 300.
    """

    def __init__(self, columns: Tuple[str, str], embeddings_size: int = 300) -> None:
        self.columns = columns
        self.embeddings_size = embeddings_size
        self.col_1_names = [f'{columns[0]}_{x}' for x in range(embeddings_size)]
        self.col_2_names = [f'{columns[1]}_{x}' for x in range(embeddings_size)]

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> EmbeddingVectorsSimilarityCalculator:
        """Fit transformer on dataframe.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            EmbeddingVectorsSimilarityCalculator: Self object.
        """
        return self

    @transformer_time_measurement_decorator('EmbeddingVectorsSimilarityCalculator')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        similarity_column = f'{self.columns[0]}_{self.columns[1]}_cosim'
        df[similarity_column] = df.apply(self.get_similarity_for_sample, axis=1)
        return df

    def get_similarity_for_sample(self, row: pd.Series) -> float:
        """Get similarity of two columns for one row.

        Args:
            row (pd.Series): Row with embedding vectors for both columns.

        Returns:
            float: Cosine similarity between two embedding vectors of two columns.
        """
        return cosine_similarity(row[self.col_1_names].values.reshape(1, -1),
                                 row[self.col_2_names].values.reshape(1, -1))[0][0]
