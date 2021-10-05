import random

import numpy as np
from util import dist
from solverInterface import Solver, Output, Instance
from visualization.clusterings import clusters


# TODO: implement probabilistic seeding
def seed(instance: Instance) -> Output:
    """
    Choose the first assignment of centers.
    As of now, the centers are chosen randomly.
    @type instance: instance of the k-median problem
    """
    centers = random.sample(instance.points, instance.k)
    return Output(instance, centers=centers)

# TODO: fix bug where a non-included point gets selected
def lloyd_iteration(assignment: Output) -> Output:
    """
    Execute one iteration of Lloyd's algorithm.
    @type output: feasible solution of the k-median problem (NOT necessarily optimal)
    """
    closest_centers = clusters(assignment.instance.points, assignment.centers)
    k = len(assignment.centers)

    # initialization of new variables
    new_centers = np.zeros((k, 2))

    # TODO: define different functions for computing medoids (e.g. via centroid, bruteforce, near-linear from paper)
    # finding the centroid of each Voronoi cell and making a new center as close as possible to the centroid

    # TODO: write this in the following way:
    #new_centers = [ medoid(cluster) for cluster in clusters ]

    for i in range(k):
        clusters_indexes = [center[0] for center in enumerate(closest_centers) if center[1] == i]
        if clusters_indexes:
            cluster_points_x = [assignment.instance.points[point][0] for point in clusters_indexes]
            cluster_points_y = [assignment.instance.points[point][1] for point in clusters_indexes]
            centroid = [np.average(cluster_points_x), np.average(cluster_points_y)]
            dists = [dist(assignment.instance.points[index], centroid) for index in clusters_indexes]
            new_center_index = np.argmin(dists)
            new_centers[i][:] = assignment.instance.points[new_center_index]
        else:
            continue

    return Output(assignment.instance, centers=new_centers)


class KMedPlusPlus(Solver):
    def solve(self, instance: Instance, numiter=100) -> Output:
        """
        Solve a k-median problem with the k-median++ algorithm
        @type instance: instance of the k-median problem
        :param numiter: number of iterations
        """
        solution = seed(instance)
        for _ in range(numiter):
            solution = lloyd_iteration(solution)
        return solution
