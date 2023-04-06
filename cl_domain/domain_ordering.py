from typing import *
import numpy as np

from cl_domain.domain import Domain
from cl_domain.utils import GLOBAL_RAND
from cl_domain.domain_similarity import EmbeddingFn, Clusterer, \
    DomainSimilarityMetric
from itertools import permutations


def path_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    return total_distance


def tsp_bruteforce(distance_matrix, start_node=0):
    n = len(distance_matrix)
    nodes = list(range(n))
    nodes.remove(start_node)

    min_distance = float('inf')
    min_path = None

    for perm in permutations(nodes):
        path = (start_node,) + perm + (start_node,)
        distance = path_distance(path, distance_matrix)

        if distance < min_distance:
            min_distance = distance
            min_path = path

    return min_distance, min_path


def random_ordering(domains: Dict[Text, Domain]) -> List[Domain]:
    """Generate a random ordering of domains."""
    domains = list(domains.values())
    GLOBAL_RAND.shuffle(domains)
    return domains


def max_path_ordering(domains: Dict[Text, str]) -> List[int]:
    """Generate a domain ordering that maximizes the number of paths between
    domains.
    """
    domain_keys = list(domains.keys())[:10]
    domain_values = list(domains.values())[:10]
    embedding_fn = EmbeddingFn()
    clustering_fn = Clusterer()
    distance_fn = DomainSimilarityMetric()
    embedding_per_domain = embedding_fn(domain_values)
    centroid_per_domain = clustering_fn(embedding_per_domain)
    distance_matrix = -distance_fn(centroid_per_domain)
    # distance_matrix = np.random.rand(10, 10)
    # distance_matrix = np.triu(distance_matrix) + np.triu(distance_matrix, k=1).T
    # np.fill_diagonal(distance_matrix, 0)

    max_distance, max_path = tsp_bruteforce(distance_matrix)
    return [domain_keys[idx] for idx in max_path[:-1]]

    # domain_distance_matrix: np.ndarray = ClusteringMetric.similarity(domain_values)

    return domains


def min_path_ordering(domains: Dict[Text, str]) -> List[int]:
    domain_keys = list(domains.keys())[:10]
    domain_values = list(domains.values())[:10]
    embedding_fn = EmbeddingFn()
    clustering_fn = Clusterer()
    distance_fn = DomainSimilarityMetric()
    embedding_per_domain = embedding_fn(domain_values)
    centroid_per_domain = clustering_fn(embedding_per_domain)
    distance_matrix = distance_fn(centroid_per_domain)
    # distance_matrix = np.random.rand(10, 10)
    # distance_matrix = np.triu(distance_matrix) + np.triu(distance_matrix, k=1).T
    # np.fill_diagonal(distance_matrix, 0)

    min_distance, min_path = tsp_bruteforce(distance_matrix)
    return [domain_keys[idx] for idx in min_path[:-1]]


ORDERINGS = {
    "random": random_ordering,
    "max_path": max_path_ordering,
    "min_path": min_path_ordering,
}