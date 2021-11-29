from __future__ import \
    annotations  # from algorithms.iterative_mistake_minimization import ClusterNode

from dataclasses import dataclass, field
from math import inf
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple
from util import Point, median_coordinatewise


@dataclass
class Instance:
    """represents an instance of the k-median problem
    assume all instances are valid d-dimensional euclidean instances"""
    points: list[Point]  # each point is np.array
    k: int  # number of centers to be opened

    def dimension(self):
        return len(self.points[0].coordinates)


@dataclass
class CenterOutput:
    """represents a solution to the k-median problem"""
    instance: Instance
    centers: list[Point]
    assignment: dict[Point, tuple[Point, np.float64]] = field(init=False)
    cost: np.float64 = field(init=False)

    def __post_init__(self):
        self.assignment = {point: point.closest_center(self.centers) for point in self.instance.points}
        self.cost = sum([self.assignment[point][1] for point in self.instance.points])

    def clusters(self) -> dict[Point, list[Point]]:
        return {center: [point for point in self.instance.points if self.assignment[point][0] == center] for center in
                self.centers}


def mistake(point: Point, center: Point, i, theta) -> bool:
    return (point.coordinates[i] <= theta) != (center.coordinates[i] <= theta)


@dataclass
class ClusterNode:
    """represents a node in an explainable tree solution to the k-median problem"""
    clusters: dict[Point, list[Point]]
    bounds: ndarray  # array of arrays (lower_bound, upper_bound) for each dimension
    set: list[Point] = field(default_factory=list) #list of centers belonging to node, default value []
    split: Optional[Tuple[int, float]] = None
    children: list[ClusterNode] = field(default_factory=list)  # default value []

    def centers(self):
        return self.clusters.keys()

    def dimension(self):
        return len(self.bounds)

    def is_homogeneous(self):
        return len(self.clusters.keys()) == 1

    def find_split(self) -> Tuple[int, float, ClusterNode, ClusterNode]:
        def count_mistakes(i, theta):
            center_point_pairs = [(center, point) for center in self.centers() for point in self.clusters[center]]
            return sum(mistake(point, center, i, theta) for center, point in center_point_pairs)

        def find_best_split_dim(i):
            center_coords = [center.coordinates[i] for center in self.centers()]
            l_i, r_i = min(center_coords), max(center_coords)

            # iterate over all potential thetas
            point_coords = [point.coordinates[i] for l in self.clusters.values() for point in l if
                            l_i <= point.coordinates[i] <= r_i]
            point_coords.sort()
            # take the midpoint of consecutive coordinates to avoid equality issues
            theta_candidates = [(a + b) / 2.0 for a, b in
                                zip(point_coords, point_coords[1:])]  # 2.0 to avoid integer division
            # TODO: implement the efficient way of counting mistakes while iterating over thetas
            brute_force_compute = [(count_mistakes(i,theta), theta) for theta in theta_candidates]
            min_mistakes, best_theta = min(brute_force_compute, key=lambda entry: entry[0])
            return min_mistakes, i, best_theta

        # def find_best_split_dim_efficient(i, point_coords, mu1, mu2, cost):
        #     print("inside efficient splitting")
        #     best_cost = inf
        #     best_threshold = None
        #
        #     # print(len(point_coords))
        #
        #     # mu1 = [medoid_bruteforce(point_coords[:j]) for j in range(1, len(point_coords))]
        #     # mu2 = [medoid_bruteforce(point_coords[j:]) for j in range(len(point_coords))]
        #
        #
        #     for j in range(len(point_coords)-1):
        #         print("j", j)
        #         cost = cost + dist(point_coords[j], mu1[j]) - dist(point_coords[j], mu2[j-1])
        #         if cost < best_cost and point_coords[j].coordinates[i] != point_coords[j+1].coordinates[i]:
        #             best_cost = cost
        #             best_threshold = point_coords[j].coordinates[i]
        #     return best_cost, i, best_threshold

        # some pre-calculations for dynamic programming
        # point_coords = [point for l in self.clusters.values() for point in l]
        # mu1 = [Point([np.median([point.coordinates[0] for point in point_coords[:j]]), \
        #               np.median([point.coordinates[1] for point in point_coords[:j]])]) \
        #        for j in range(1, len(point_coords))]
        # mu2 = [Point([np.median([point.coordinates[0] for point in point_coords[j:]]), \
        #               np.median([point.coordinates[1] for point in point_coords[j:]])]) \
        #        for j in range(len(point_coords))]
        # cost = sum([dist(point, mu2[0]) for point in point_coords])
        # find_best_split_dim_efficient(i, point_coords, mu1, mu2, cost)

        # compute best splits in each dimension
        split_candidates = [find_best_split_dim(i) for i in range(self.dimension())]
        _, i, theta = min(split_candidates, key=lambda entry: entry[0])
        node_L, node_R = make_kids(self, i, theta)
        return i, theta, node_L, node_R

def make_kids(node: ClusterNode, i: int, theta: float, useEsfandiari = False):
    # update clusters and bounds for children nodes
    node_L_centers = [center for center in node.centers() if center.coordinates[i] <= theta]
    node_L_clusters = {center: [point for point in node.clusters[center] if point.coordinates[i] <= theta] for
                       center in node_L_centers}
    if useEsfandiari:
        node_L_X = node_L_centers
    else:
        node_L_X = node_L_centers + list(node_L_clusters.values())
    node_L_bounds = node.bounds.copy()
    node_L_bounds[i][1] = theta  # change upper bound to theta
    node_L = ClusterNode(node_L_clusters, node_L_bounds, node_L_X)

    node_R_centers = [center for center in node.centers() if center.coordinates[i] > theta]
    node_R_clusters = {center: [point for point in node.clusters[center] if point.coordinates[i] > theta] for
                       center in node_R_centers}
    if useEsfandiari:
        node_R_X = node_R_centers
    else:
        node_R_X = node_R_centers + list(node_R_clusters.values())
    node_R_bounds = node.bounds.copy()
    node_R_bounds[i][0] = theta  # change lower bound to theta
    node_R = ClusterNode(node_R_clusters, node_R_bounds, node_R_X)
    return node_L, node_R

@dataclass
class ExplainableOutput:
    """represents an explainable solution to the k-median problem"""
    instance: Instance
    leaves: list[ClusterNode]
    split_nodes: list[ClusterNode]
    pre_clusters: dict[Point, list[Point]]
    medians: Optional[list[Point]] = None
    # TODO: add field pre_clusters, refactor

    def __post_init__(self):
        self.clusters = self.clusters()

    def clusters(self) -> dict[int, list[Point]]:
        self.medians = [median_coordinatewise(list(node.clusters.values())[0]) for node in
                        self.leaves]  # for each leaf node, get list of points in (only) cluster
        assignment = {point: point.closest_center(self.medians) for point in self.instance.points}
        return {center: [point for point in self.instance.points if assignment[point][0] == center] for center in
                self.medians}
