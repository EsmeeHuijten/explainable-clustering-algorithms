import random
import numpy as np
from solverInterface import Instance
from functools import reduce


def k_clusters(k: int, cluster_size: int = 10, spread: float = 0.3) -> Instance:
    '''
    returns an instance of k-th roots of unity with cluster_size many points around them
    '''
    aux_poly = [1] + [0] * (k - 1) + [-1]
    roots = np.roots(aux_poly)
    centers = [np.array([root.real, root.imag]) for root in roots]

    def perturb(center, cluster_size, spread):
        perturbed_points = [np.array([coord + spread * random.random() for coord in center]) for _ in
                            range(cluster_size)]
        return perturbed_points + [center]

    nested_points = [perturb(center, cluster_size, spread) for center in centers]
    points = reduce(list.__add__, nested_points)
    return Instance(points, k)
