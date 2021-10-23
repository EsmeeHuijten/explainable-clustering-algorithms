import random
from functools import reduce
import numpy as np
from solver_interface import Instance


# TODO: implement other synthetic clusterings, e.g.  centers in star and points with exp distribution
from util import Point


def k_clusters(k: int, cluster_size: int = 10, spread: float = 0.3, instance_seed: int = 0) -> Instance:
    """
    Calculates an instance of the k-median problem of k-th roots of unity with cluster_size many points around them.
    @param k:  the number of centers
    @param cluster_size: (optional) the size (number of datapoints) of each cluster
    @param spread: (optional) the spread of each cluster
    @param instance_seed: (optional) seeding of the instance
    @return: an instance of the k-median problem of k-th roots of unity with cluster_size many points around them
    """
    random.seed(instance_seed)
    aux_poly = [1] + [0] * (k - 1) + [-1]
    roots = np.roots(aux_poly)
    centers = [Point(np.array([root.real, root.imag])) for root in roots]

    def perturb(center):
        """
        Computes cluster_size perturbed points to form a cluster.
        @param center: the center of the cluster to be formed
        @return: list of perturbed points that form a cluster
        """
        perturbed_points = [Point(np.array([coord + spread * random.random() for coord in center.coordinates])) for _ in
                            range(cluster_size)]
        return perturbed_points + [center]

    nested_points = [perturb(center) for center in centers]
    points = reduce(list.__add__, nested_points)
    return Instance(points, k)
