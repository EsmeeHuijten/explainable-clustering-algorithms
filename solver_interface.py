from __future__ import \
    annotations  # allows us to use ClusterNode in type hints of ClusterNode methods. this will be default in Python 3.10

from dataclasses import dataclass, field
from anytree import Node, RenderTree

import numpy as np
from numpy import ndarray
from typing import Optional, Tuple
# from algorithms.iterative_mistake_minimization import ClusterNode
from util import Point, medoid_bruteforce


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
    # TODO: how to implement Output? centers, clusters, decision tree? Separate class for explainable clusterings?
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
    split: Optional[Tuple[int, float]] = None
    children: list[ClusterNode] = field(default_factory=list)  # needed in order to get default value []

    def centers(self):
        return self.clusters.keys()

    def dimension(self):
        return len(list(self.clusters.keys())[0].coordinates)

    def is_homogeneous(self):
        return len(self.clusters.keys()) == 1

    def find_split(self) -> Tuple[int, float, ClusterNode, ClusterNode]:
        def count_mistakes(i, theta):
            center_point_pairs = [(center, point) for center in self.centers() for point in self.clusters[center]]
            return sum(mistake(point, center, i, theta) for center, point in center_point_pairs), theta

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
            brute_force_compute = [count_mistakes(i, theta) for theta in theta_candidates]
            min_mistakes, best_theta = min(brute_force_compute, key=lambda entry: entry[0])
            return min_mistakes, i, best_theta

        # compute best splits in each dimension
        split_candidates = [find_best_split_dim(i) for i in range(self.dimension())]
        _, i, theta = min(split_candidates, key=lambda entry: entry[0])

        # update clusters and bounds for children nodes
        node_L_centers = [center for center in self.centers() if center.coordinates[i] <= theta]
        node_L_clusters = {center: [point for point in self.clusters[center] if point.coordinates[i] <= theta] for
                           center in node_L_centers}
        node_L_bounds = self.bounds.copy()
        node_L_bounds[i][1] = theta  # change upper bound to theta
        node_L = ClusterNode(node_L_clusters, node_L_bounds)

        node_R_centers = [center for center in self.centers() if center.coordinates[i] > theta]
        node_R_clusters = {center: [point for point in self.clusters[center] if point.coordinates[i] > theta] for
                           center in node_R_centers}
        node_R_bounds = self.bounds.copy()
        node_R_bounds[i][0] = theta  # change lower bound to theta
        node_R = ClusterNode(node_R_clusters, node_R_bounds)
        return i, theta, node_L, node_R

@dataclass
class ExplainableOutput:
    """represents an explainable solution to the k-median problem"""
    instance: Instance
    leaves: list[ClusterNode]
    split_nodes: list[ClusterNode]
    medoids: Optional[list[Point]] = None

    def __post_init__(self):
        self.clusters = self.clusters()

    # # TODO: after algorithm returns tree, compute medoids and return clusters as dict[Point, list[Point]] as output type
    def clusters(self) -> dict[int, list[Point]]:
        self.medoids = [medoid_bruteforce(list(node.clusters.values())[0]) for node in
                   self.leaves]  # for each leaf node, get list of points in (only) cluster
        assignment = {point: point.closest_center(self.medoids) for point in self.instance.points}
        return {center: [point for point in self.instance.points if assignment[point][0] == center] for center in
                self.medoids}