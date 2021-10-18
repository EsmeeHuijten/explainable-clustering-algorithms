"""
implements the k-means++ algorithm for the k-median objective
"""
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from solver_interface import Output, Instance
from util import dist, Point


def random_seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers.
    As of now, the centers are chosen randomly.
    @param instance: instance of the k-median problem
    """
    centers = random.sample(instance.points, instance.k)
    return Output(instance, centers=centers)


def prob_seed(instance: Instance, seed: Optional[int]) -> Output:
    """
    Choose the first assignment of centers using probabilistic seeding.
    @param instance: instance of the k-median problem
    @param seed: optional seed for random choices
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
    return Output(instance, centers=centers)


def closest_to_centroid(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster via clusters_indexes, this function computes the centroid of that cluster
    and returns the point closest to the centroid.
    @param clusterpoints: all points of cluster in question
    """
    cluster_points_x = [point.coordinates[0] for point in clusterpoints]
    cluster_points_y = [point.coordinates[1] for point in clusterpoints]
    centroid = Point((np.average(cluster_points_x), np.average(cluster_points_y)))
    dists = [dist(point, centroid) for point in clusterpoints]
    new_center = clusterpoints[np.argmin(dists)]
    return new_center


def medoid_bruteforce(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster via clusters_indexes, this function computes the medoid of that cluster
    and returns the medoid as well as the "cost" of the medoid, which is the sum of distances from each
    point in the cluster to the medoid. This function calculates the medoid with the brute force method.
    @param clusterpoints: points of cluster in question
    """
    cost = [sum([dist(point1, point2) for point2 in clusterpoints]) for point1 in clusterpoints]
    return clusterpoints[np.argmin(cost)]


def lloyd_iteration(assignment: Output) -> Output:
    """
    Execute one iteration of Lloyd's algorithm.
    @param assignment: current assignment of centers
    @param return: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    clusterassignment = assignment.clusters()
    new_centers = [medoid_bruteforce(clusterpoints) for center, clusterpoints
                   in clusterassignment.items() if clusterpoints]
    return Output(assignment.instance, centers=new_centers)


@dataclass
class KMedPlusPlus:
    numiter: int = 1
    seed: Optional[int] = 0
    visualize: bool = False

    def __call__(self, instance: Instance) -> Output:
        """
        Solve a k-median problem with the k-median++ algorithm
        @param instance: instance of the k-median problem
        """
        solution = prob_seed(instance, self.seed)
        for _ in range(self.numiter):
            solution = lloyd_iteration(solution)
            if self.visualize:
                visualization.clusterings.show_clusters(solution)
        return solution
