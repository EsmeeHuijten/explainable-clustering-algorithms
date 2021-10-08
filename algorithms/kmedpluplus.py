import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from util import dist
from solverInterface import Output, Instance
from visualization.clusterings import clusters


def random_seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers.
    As of now, the centers are chosen randomly.
    @type instance: instance of the k-median problem
    """
    centers = random.sample(instance.points, instance.k)
    return Output(instance, centers=centers)


def prob_seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers using probabilistic seeding.
    @type instance: instance of the k-median problem
    """
    centers = np.zeros(instance.k)
    centers[0] = random.choice(instance.points)
    for i in range(instance.k - 1):
        distances_squared = [dist_to_nearest_center(centers[0:i + 1], point) ** 2 for point in instance.points]
        dists_sq_norm = distances_squared / sum(distances_squared)
        new_index = random.choice(np.arange(0, len(instance.points)), p=dists_sq_norm)
        centers[i + 1] = instance.points[int(new_index)]


def dist_to_nearest_center(centers, point):
    distances = [dist(centers[i], point) for i in range(len(centers))]
    return min(distances)


def closest_to_centroid(assignment: Output, clusters_indexes):
    """
    Given a cluster via clusters_indexes, this function computes the centroid of that cluster
    and returns the point closest to the centroid.
    @param assignment: current assignment of centers
    @param clusters_indexes: indexes of the points of cluster in question
    @type output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    cluster_points_x = [assignment.instance.points[point][0] for point in clusters_indexes]
    cluster_points_y = [assignment.instance.points[point][1] for point in clusters_indexes]
    centroid = [np.average(cluster_points_x), np.average(cluster_points_y)]
    dists = [dist(assignment.instance.points[index], centroid) for index in clusters_indexes]
    new_center_index = np.argmin(dists)
    return new_center_index


def medoid_bruteforce(assignment: Output, clusters_indexes):
    """
    Given a cluster via clusters_indexes, this function computes the medoid of that cluster
    and returns the medoid as well as the "cost" of the medoid, which is the sum of distances from each
    point in the cluster to the medoid. This function calculates the medoid with the brute force method.
    @param assignment: current assignment of centers
    @param clusters_indexes: indexes of the points of cluster in question
    @type Output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    cost = [sum([dist(point, point2) for point2 in assignment.instance.points]) for point in assignment.instance.points]
    return np.argmin(cost), min(cost)  # index of medoid, cost of medoid


# TODO: fix bug where a non-included point gets selected
def lloyd_iteration(assignment: Output) -> Output:
    """
    Execute one iteration of Lloyd's algorithm.
    @param assignment: current assignment of centers
    @type Output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    clusterassignment = assignment.clusters()
    print("clusterassignment", clusterassignment)
    for name, indexes in clusterassignment.items():
        print(name, indexes)
    new_centers = [closest_to_centroid(assignment, indexes) for name, indexes in clusterassignment.items() if indexes]
    print("new_centers", [center for center in new_centers])
    # closest_centers = clusters(assignment.instance.points, assignment.centers)
    # k = len(assignment.centers)

    # initialization of new variables
    # new_centers = np.zeros((k, 2))

    # TODO: define different functions for computing medoids (e.g. via centroid, bruteforce, near-linear from paper)
    # finding the centroid of each Voronoi cell and making a new center as close as possible to the centroid

    # TODO: write this in the following way:
    # new_centers = [ medoid(cluster) for cluster in clusters ]

    # for i in range(k):
    #     clusters_indexes = [center[0] for center in enumerate(closest_centers) if center[1] == i]
    #     if clusters_indexes:
    #         new_center_index = compute_centroid_center(assignment, clusters_indexes)
    #         new_centers[i][:] = assignment.instance.points[new_center_index]
    #     else:
    #         continue

    return Output(assignment.instance, centers=new_centers)

@dataclass
class KMedPlusPlus:
    numiter: int = 3
    def __call__(self, instance: Instance) -> Output:
        """
        Solve a k-median problem with the k-median++ algorithm
        @type instance: instance of the k-median problem
        :param numiter: number of iterations
        """
        solution = random_seed(instance)
        for _ in range(self.numiter):
            solution = lloyd_iteration(solution)
        return solution
