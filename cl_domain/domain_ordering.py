from typing import *
import numpy as np
import pandas as pd

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


def max_path_ordering(domains: Dict[Text, str]) -> List[str]:
    """Generate a domain ordering that maximizes the number of paths between
    domains.
    """
    df = pd.read_csv('./distance_matrix.csv', index_col=0)
    row_col_to_drop = set(df.columns) - set(domains.keys())
    df = df.drop(row_col_to_drop, axis=1).drop(row_col_to_drop, axis=0)
    domain_keys = df.columns.tolist()
    distance_matrix = -df.iloc[:, :].values
    max_distance, max_path = tsp_bruteforce(distance_matrix)
    return [domain_keys[idx] for idx in max_path[:-1]]


def min_path_ordering(domains: Dict[Text, str]) -> List[str]:
    df = pd.read_csv('./distance_matrix.csv', index_col=0)
    row_col_to_drop = set(df.columns) - set(domains.keys())
    df = df.drop(row_col_to_drop, axis=1).drop(row_col_to_drop, axis=0)
    domain_keys = df.columns.tolist()
    distance_matrix = df.iloc[:, :].values
    min_distance, min_path = tsp_bruteforce(distance_matrix)
    return [domain_keys[idx] for idx in min_path[:-1]]


ORDERINGS = {
    "random": random_ordering,
    "max_path": max_path_ordering,
    "min_path": min_path_ordering,
}