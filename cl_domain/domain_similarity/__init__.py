from dataclasses import dataclass
from typing import *
import numpy as np

from cl_domain.domain import Domain
from cl_domain.domain_similarity.embedding import EMBEDDING
from cl_domain.domain_similarity.clusterer import CLUSTERER
from cl_domain.domain_similarity.distance import DISTANCE


class DomainSimilarityMetric(object):
    """Base class for data domain_similarity metrics."""

    def __call__(self, centroids_per_domain: List[List[float]]) -> np.ndarray:
        distance_fn = DISTANCE['eud']
        distance_matrix = np.zeros((len(centroids_per_domain), len(centroids_per_domain)), dtype=float)
        for i in range(len(centroids_per_domain)):
            for j in range(len(centroids_per_domain)):
                distance_matrix[i][j] = distance_fn(centroids_per_domain[i], centroids_per_domain[j])
        
        return distance_matrix



class Clusterer(object):
    """Base class for clustering algorithms."""

    def __call__(self, embedding_per_domain: List[List[List[float]]]) -> List[List[float]]:
        clusterer_fn = CLUSTERER['regular']
        centroids_per_domain = clusterer_fn(embedding_per_domain)
        return centroids_per_domain


class EmbeddingFn(object):
    """Base class for embedding functions."""

    def __call__(self, data_per_domain: List[List[str]]) -> List[List[List[float]]]:
        embedding_fn = EMBEDDING['sbert']
        embeddings_per_domain = embedding_fn(data_per_domain)
        return embeddings_per_domain


# @dataclass(frozen=True)
# class ClusteringMetric(DomainSimilarityMetric):
#     """Common class for data domain_similarity metrics that use embedding + clustering methods."""

#     embedding_fn: EmbeddingFn
#     clusterer: Clusterer
#     cluster_similarity_fn: Callable

#     def similarity(self, domain_values: List) -> np.ndarray:
#         similarities = []
#         print(len(domain_values))
#         # embeddings_a = self.embedding_fn(domain_a)
#         # embeddings_b = self.embedding_fn(domain_b)
#         # clusters_a = self.clusterer(embeddings_a)
#         # clusters_b = self.clusterer(embeddings_b)
#         print('Hi')
#         return None #self.cluster_similarity_fn(clusters_a, clusters_b)
