from typing import *

from cl_domain.domain import Domain
from cl_domain.utils import GLOBAL_RAND


def tsne_clustering(embedding_per_domain: List[List[List[float]]]) -> List[List[float]]:
    raise NotImplementedError


def regular_clustering(embedding_per_domain: List[List[List[float]]]) -> List[List[float]]:
    """Generate a random ordering of domains."""
    centroids_per_domain = []
    for embedding in embedding_per_domain:
        centroids = [sum(x) for x in zip(*embedding)]
        centroids_per_domain.append(centroids)

    return centroids_per_domain


CLUSTERER = {
    "tsne": tsne_clustering,
    "regular": regular_clustering
}
