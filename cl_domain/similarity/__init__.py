from dataclasses import dataclass
from typing import *
import numpy as np

from cl_domain.data import Domain


class DomainSimilarityMetric(object):
    """Base class for data similarity metrics."""

    def similarity(self, domain_a: Domain, domain_b: Domain) -> float:
        raise NotImplementedError


class Clusterer(object):
    """Base class for clustering algorithms."""

    def __call__(self, embeddings: np.ndarray) -> List[List[int]]:
        raise NotImplementedError


class EmbeddingFn(object):
    """Base class for embedding functions."""

    def __call__(self, domain: Domain) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class ClusteringMetric(DomainSimilarityMetric):
    """Common class for data similarity metrics that use embedding + clustering methods."""

    embedding_fn: EmbeddingFn
    clusterer: Clusterer
    cluster_similarity_fn: Callable

    def similarity(self, domain_a: Domain, domain_b: Domain) -> float:
        embeddings_a = self.embedding_fn(domain_a)
        embeddings_b = self.embedding_fn(domain_b)
        clusters_a = self.clusterer(embeddings_a)
        clusters_b = self.clusterer(embeddings_b)
        return self.cluster_similarity_fn(clusters_a, clusters_b)
