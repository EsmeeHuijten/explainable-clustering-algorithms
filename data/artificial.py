import numpy as np

from solverInterface import Instance


# TODO: implement k_clusters, returning 2D-data with k artificial clusters
def k_clusters(k: int) -> Instance:
    points = np.array([])
    return Instance(points, k)


def toy_input(n, k):
    points = [np.array([i]) for i in range(n)]
    return Instance(points, k)
