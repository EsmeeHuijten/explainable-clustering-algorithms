# TODO: implement k-median++; i.e. initial probabilistic seeding with weights and LLoyd's iterative improvement

import numpy as np
import random
import sys
sys.path.append('../')

from solverInterface import Solver, Output, Instance
from visualization.clusterings import clusters, eucl_dist

# TODO: make structure nicer with __init__ function
class KMedian(Solver):
    def solve(self, instance: Instance) -> Output:
        """
        Solve a k-median problem with the k-median++ algorithm
        @type instance: instance of the k-median problem
        """
        numiter = 100
        solution = self.seed(instance)
        for i in range(numiter):
            solution = self.iteration(solution)
        return solution


    def seed(self, instance: Instance) -> Output:
        """
        Choose the first assignment of centers.
        As of now, the centers are chosen randomly.
        @type instance: instance of the k-median problem
        """
        centers = random.sample(instance.points, instance.k)
        return Output(instance, centers=centers, clusters=[])

# TODO: add the right clusters to the returned output
    def iteration(self, assignment: Output) -> Output:
        """
        Execute one iteration of Lloyd's algorithm.
        @type output: feasible solution of the k-median problem (NOT necessarily optimal)
        """
        closest_centers, cost = clusters(assignment.instance.points, assignment.centers)
        k = len(assignment.centers)

        # initialization of new variables
        new_centers = np.zeros((k, 2))

        # finding the centroid of each Voronoi cell and making a new center as close as possible to the centroid
        for i in range(k):
            clusters_indexes = [center[0] for center in enumerate(closest_centers) if center[1] == i]
            if clusters_indexes:
                cluster_points_x = [assignment.instance.points[point][0] for point in clusters_indexes]
                cluster_points_y = [assignment.instance.points[point][1] for point in clusters_indexes]
                centroid = [np.average(cluster_points_x), np.average(cluster_points_y)]
                dists = [eucl_dist(assignment.instance.points[index], centroid) for index in clusters_indexes]
                new_center_index = np.argmin(dists)
                new_centers[i][:] = assignment.instance.points[new_center_index]
            else:
                continue

        return Output(assignment.instance, centers=new_centers, clusters=[])