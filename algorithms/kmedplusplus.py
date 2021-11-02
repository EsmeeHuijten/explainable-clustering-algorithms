"""
implements the k-means++ algorithm for the k-median objective
"""
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from solver_interface import CenterOutput, Instance
from util import dist, Point


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


def closest_to_centroid(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster, this function computes the centroid of that cluster
    and returns the point closest to the centroid.
    @param clusterpoints: all points of cluster in question
    @return: the datapoint closest to the centroid of the cluster in question
    """
    cluster_points_x = [point.coordinates[0] for point in clusterpoints]
    cluster_points_y = [point.coordinates[1] for point in clusterpoints]
    centroid = Point((np.average(cluster_points_x), np.average(cluster_points_y)))
    dists = [dist(point, centroid) for point in clusterpoints]
    new_center = clusterpoints[np.argmin(dists)]
    return new_center


def medoid_bruteforce(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster, this function computes the medoid of that cluster
    and returns it. This function calculates the medoid with the brute force method.
    @param clusterpoints: points of cluster in question
    @return: medoid of the cluster in question
    """
    cost = [sum([dist(point1, point2) for point2 in clusterpoints]) for point1 in clusterpoints]
    return clusterpoints[np.argmin(cost)]


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
