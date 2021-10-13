import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from util import dist, Point, closest_center
from solverInterface import Output, Instance


def random_seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers.
    As of now, the centers are chosen randomly.
    @type instance: instance of the k-median problem
    """
    centers = random.sample(instance.points, instance.k)
    return Output(instance, centers=centers)

# TODO: construct centers array without initialization (foldleft?)
def prob_seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers using probabilistic seeding.
    @type instance: instance of the k-median problem
    """
    centers = [Point((0, 0))] * instance.k
    centers[0] = np.random.choice(instance.points)
    for i in range(instance.k - 1):
        distances_sq = [closest_center(point, centers[0:i + 1])[1] ** 2 for point in instance.points]
        dists_sq_norm = distances_sq / sum(distances_sq)
        new_index = np.random.choice(np.arange(0, len(instance.points)), p=dists_sq_norm) #np.random method needed for p argument!
        centers[i + 1] = instance.points[int(new_index)]
    return Output(instance, centers=centers)

def dist_to_nearest_center(centers, point):
    distances = [dist(centers[i], point) for i in range(len(centers))]
    return min(distances)


def closest_to_centroid(assignment: Output, clusterpoints):
    """
    Given a cluster via clusters_indexes, this function computes the centroid of that cluster
    and returns the point closest to the centroid.
    @param assignment: current assignment of centers
    @param clusterpoints: all points of cluster in question
    @type output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    cluster_points_x = [point.coordinates[0] for point in clusterpoints]
    cluster_points_y = [point.coordinates[1] for point in clusterpoints]
    centroid = Point((np.average(cluster_points_x), np.average(cluster_points_y)))
    dists = [dist(point, centroid) for point in clusterpoints]
    new_center = clusterpoints[np.argmin(dists)]
    return new_center


def medoid_bruteforce(assignment: Output, clusterpoints):
    """
    Given a cluster via clusters_indexes, this function computes the medoid of that cluster
    and returns the medoid as well as the "cost" of the medoid, which is the sum of distances from each
    point in the cluster to the medoid. This function calculates the medoid with the brute force method.
    @param assignment: current assignment of centers
    @param clusters_indexes: indexes of the points of cluster in question
    @type Output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    cost = [sum([dist(point1, point2) for point2 in clusterpoints]) for point1 in clusterpoints]
    return clusterpoints[np.argmin(cost)]


# TODO: fix bug where a non-included point gets selected
def lloyd_iteration(assignment: Output) -> Output:
    """
    Execute one iteration of Lloyd's algorithm.
    @param assignment: current assignment of centers
    @type Output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    clusterassignment = assignment.clusters()
    new_centers = [medoid_bruteforce(assignment, clusterpoints) for center, clusterpoints \
                   in clusterassignment.items() if clusterpoints]
    return Output(assignment.instance, centers=new_centers)

@dataclass
class KMedPlusPlus:
    numiter: int = 1
    def __call__(self, instance: Instance) -> Output:
        """
        Solve a k-median problem with the k-median++ algorithm
        @type instance: instance of the k-median problem
        :param numiter: number of iterations
        """
        solution = prob_seed(instance)
        for _ in range(self.numiter):
            solution = lloyd_iteration(solution)
        return solution
