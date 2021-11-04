"""
implements the k-means++ algorithm for the k-median objective
"""
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import visualization.clusterings

from solver_interface import CenterOutput, Instance
from util import medoid_bruteforce


def random_seed(instance: Instance) -> CenterOutput:
    """
    Choose the first assignment of centers randomly.
    @param instance: instance of the k-median problem
    @return: seeded solution (Output instance)
    """
    centers = random.sample(instance.points, instance.k)
    return CenterOutput(instance, centers=centers)


def prob_seed(instance: Instance, seed: Optional[int]) -> CenterOutput:
    """
    Choose the first assignment of centers using probabilistic seeding.
    @param instance: instance of the k-median problem
    @param seed: optional seed for random choices
    @return: seeded solution (Output instance)
    """
    if seed is not None:
        np.random.seed(seed)
    centers = [np.random.choice(instance.points)]  # choose first center randomly
    for i in range(1, instance.k):
        distances_sq = np.array([point.closest_center(centers[0:i + 1])[1] ** 2 for point in instance.points])
        dists_sq_norm = distances_sq / sum(distances_sq)
        new_index = np.random.choice(np.arange(0, len(instance.points)),
                                     p=dists_sq_norm)  # np.random method needed for p argument!
        centers.append(instance.points[int(new_index)])
    return CenterOutput(instance, centers=centers)


def lloyd_iteration(assignment: CenterOutput) -> CenterOutput:
    """
    Execute one iteration of Lloyd's algorithm.
    @param assignment: current assignment of centers
    @return: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    clusterassignment = assignment.clusters()
    new_centers = [medoid_bruteforce(clusterpoints) for center, clusterpoints
                   in clusterassignment.items() if clusterpoints]
    return CenterOutput(assignment.instance, centers=new_centers)


@dataclass
class KMedPlusPlus:
    numiter: int = 1
    seed: Optional[int] = 0
    visualize: bool = False

    def __call__(self, instance: Instance) -> CenterOutput:
        """
        Solve a k-median problem with the k-median++ algorithm
        @param instance: instance of the k-median problem
        @return: a solution to the k-median problem
        """
        solution = prob_seed(instance, self.seed)
        for _ in range(self.numiter):
            solution = lloyd_iteration(solution)
            if self.visualize:
                visualization.clusterings.show_clusters(solution)
        return solution
