from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Iterable, List
from src.helpers import load_serialized_object
from src.preprocessing.decorators import transformer_time_measurement_decorator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import homogeneity_score, silhouette_score


class Leader:
    """Clustering algorithm Leader with deriving features from created clusters.

    Leader is incremental clustering algorithm. Algorithm is passing through
    the data samples and similarity between each sample and actually existing
    clusters is calculated. Leader is a representative sample (does not have
    to be real sample from dataset) of cluster, so similarity is calculated
    between sample and leader of each cluster. If similarity of sample is higher
    than pre-defined threshold with at least one cluster, sample is added to
    cluster where the similarity is highest and leader of this cluster is
    re-calculated as a mean of all samples (including new one). If similarity
    is lower than pre-defined threshold with every existing cluster leaders,
    new cluster is created with actual sample. As a similarity metric,
    cosine similarity is used.

    Properties of algorithm:
    - each sample belongs to exactly one cluster,
    - when running the algorithm with the same data, but after shuffling,
        created clusters may be different,
    - there is no need to specify number of clusters,
    - threshold must be chosen wisely to achieve expected result,
    - clusters are possibly overlapping (even though each sample belongs to only one cluster).

    Paper:
    Hartigan, J.A.: Clustering Algorithms. John Wiley & Sons, Inc., New York,
    USA, 1975.

    Args:
        threshold (float): Pre-defined threshold of similarity that decides
            whether actual sample is added to cluster or not.
        columns (List[str], optional): Columns to be used for similarity calculation.
            Defaults to None.
        additional_data_path (str, optional): Path to additional data to be used for
            clustering. However, only labels for data from pipeline will be generated.
            Defaults to None.
        clusters (dict): Dictionary with dataframe with all samples for each cluster.
            Cluster index is a key in dictionary.
        leaders (np.ndarray): All leaders represented with only those features used
            for similarity calculation.
    """

    def __init__(self, threshold: float = 0.8, columns: List[str] = None,
                 additional_data_path: str = None) -> None:
        """Initialize new Leader object.

        Args:
            threshold (float, optional): Pre-defined threshold of similarity that
                decides whether actual sample is added to cluster or not. Defaults to 0.8.
            columns (List[str], optional): Columns to be used for similarity calculation.
                Defaults to None.
            additional_data_path (str, optional): Path to additional data to be used for
                clustering. However, only labels for data from pipeline will be generated.
                Defaults to None.
        """
        self.threshold = threshold
        self.columns = columns
        self.additional_data_path = additional_data_path
        self.clusters = {}
        self.leaders = []

    def fit(self, df: pd.DataFrame, y: Iterable = None) -> Leader:
        """Fit transformer on dataframe.

        Important note: As a part of this fit function, also original dataset is changed.
        New column with labels for train data is added - this is just a workaround to
        achieve same clusters for train data that have been derived in clustering phase.

        Args:
            df (pd.Dataframe): Dataframe to fit transformer on.
            y (Iterable): Training targets.

        Returns:
            Leader: Self object.
        """
        if self.columns is None:
            self.columns = list(df.columns)

        n_clusters = 0
        labels = []

        df_additional = load_serialized_object(self.additional_data_path) \
            if self.additional_data_path is not None else pd.DataFrame()

        n_train = df.shape[0]

        for idx, row in pd.concat([df, df_additional], ignore_index=True).iterrows():
            cluster_index = -1
            if n_clusters > 0:
                similarities = cosine_similarity(row[self.columns].values.reshape(1, -1),
                                                 self.leaders)[0]

                most_similar_cluster = np.argmax(similarities)
                if similarities[most_similar_cluster] >= self.threshold:
                    cluster_index = most_similar_cluster

            if cluster_index != -1:
                self.clusters[cluster_index] = pd.concat([self.clusters[cluster_index],
                                                          row.to_frame().T],
                                                         ignore_index=True)
                self.leaders[cluster_index] = self.clusters[cluster_index][self.columns].mean()
                self.leaders[cluster_index] = self.leaders[cluster_index].values
            else:
                cluster_index = n_clusters
                self.clusters[cluster_index] = row.to_frame().T.reset_index(drop=True)
                self.leaders.append(row[self.columns].values)
                n_clusters += 1

            if idx < n_train:
                labels.append(cluster_index)

        df['leader_cluster'] = labels

        print('>> Clustering evaluation metrics')
        print(f'  Homogeneity score: {homogeneity_score(labels, y)}')
        print(f'  Silhouette score: {silhouette_score(df[self.columns], labels)}')

        return self

    @transformer_time_measurement_decorator('Leader')
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform columns of dataframe.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        # Cluster labels were already assigned in 'fit' function for train data
        if 'leader_cluster' not in df.columns:
            df['leader_cluster'] = df.apply(self.get_cluster_for_sample, axis=1)

        all_columns = list(df.drop(columns=['leader_cluster']).columns)

        df = df.apply(lambda row: self.create_cluster_features(row, columns=all_columns), axis=1)

        return df

    def get_cluster_for_sample(self, row: pd.Series) -> int:
        """Get cluster label for given sample.

        Args:
            row (pd.Series): One row of dataframe.

        Returns:
            int: Cluster index or -1, if sample is not similar to any of leaders.
        """
        similarities = cosine_similarity(row[self.columns].values.reshape(1, -1), self.leaders)

        for cluster_index in np.argsort(similarities, axis=None)[::-1]:
            if similarities[0][cluster_index] > self.threshold:
                return cluster_index

        return -1

    def create_cluster_features(self, row: pd.Series, columns: List[str]) -> pd.Series:
        """Create features according to cluster.

        Args:
            row (pd.Series): One row to create features for.
            columns (List[str]): List of columns to create aggregated features from.

        Returns:
            pd.Series: Row with new features.
        """
        df_cluster = self.clusters[row['leader_cluster']] \
            if row['leader_cluster'] != -1 else row.to_frame().T

        avg_values = df_cluster[columns].mean().values

        for idx, column in enumerate(columns):
            row[f'cluster_avg_{column}'] = avg_values[idx]

        row['cluster_length'] = df_cluster.shape[0]

        return row
