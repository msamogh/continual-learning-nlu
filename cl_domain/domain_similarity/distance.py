from sklearn.metrics.pairwise import euclidean_distances


def euclidean_distance(a, b):
    return euclidean_distances([a], [b])[0][0]
