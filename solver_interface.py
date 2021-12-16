from __future__ import \
    annotations  # from algorithms.iterative_mistake_minimization import ClusterNode

from dataclasses import dataclass, field
from functools import reduce
from math import inf
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple
from util import Point, median_coordinatewise, dist


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

    # def cost(self) -> float:
    #     clusters = self.clusters()
    #     cost = sum([sum([dist(center, point) for point in clusters[center]]) for center in list(clusters.keys())])
    #     return cost


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


def make_kids(node: ClusterNode, i: int, theta: float, useEsfandiari = False):
    # update clusters and bounds for children nodes
    node_L_centers = [center for center in node.centers() if center.coordinates[i] <= theta]
    # node_L_clusters = {center: [point for point in node.clusters[center] if point.coordinates[i] <= theta] for
    #                    center in node_L_centers}
    # for point in list(node.clusters.values()):
    #     print(point)
    node_L_clusters = {center: [point for point in reduce(list.__add__, node.clusters.values())
                                if (point.coordinates[i] <= theta and point.closest_center(node_L_centers)[0] == center)] for
                       center in node_L_centers}
    if useEsfandiari:
        node_L_X = node_L_centers
    else:
        node_L_X = node_L_centers + reduce(list.__add__, node.clusters.values())
    node_L_bounds = node.bounds.copy()
    node_L_bounds[i][1] = theta  # change upper bound to theta
    node_L = ClusterNode(node_L_clusters, node_L_bounds, node_L_X)

    node_R_centers = [center for center in node.centers() if center.coordinates[i] > theta]
    # node_R_clusters = {center: [point for point in node.clusters[center] if point.coordinates[i] > theta] for
    #                    center in node_R_centers}
    node_R_clusters = {center: [point for point in reduce(list.__add__, node.clusters.values())
                                if (point.coordinates[i] > theta and point.closest_center(node_R_centers)[0] == center)]
                       for center in node_R_centers}
    if useEsfandiari:
        node_R_X = node_R_centers
    else:
        node_R_X = node_R_centers + reduce(list.__add__, node.clusters.values())
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

    def __post_init__(self):
        self.clusters = self.clusters()

    def clusters(self) -> dict[int, list[Point]]:
        self.medians = [median_coordinatewise(node.set) for node in
                        self.leaves]  # for each leaf node, get list of points in (only) cluster
        assignment = {point: point.closest_center(self.medians) for point in self.instance.points}
        return {center: [point for point in self.instance.points if assignment[point][0] == center] for center in
                self.medians}

    def cost(self) -> float:
        cost = sum([sum([dist(center, point) for point in self.clusters[center]]) for center in list(self.clusters.keys())])
        return cost
