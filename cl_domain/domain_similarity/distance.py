from sklearn.metrics.pairwise import euclidean_distances
from typing import *
import math


def euclidean_distance(x: List[float], y: List[float]) -> float:
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


DISTANCE = {
    "eud": euclidean_distance
}
