from __future__ import \
    annotations  # allows us to use ClusterNode in type hints, which will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf

import numpy as np
from typing import Tuple

import algorithms.kmedplusplus
from solver_interface import Instance, ExplainableOutput, ClusterNode, make_kids
from util import Point

@dataclass
class IMM:
    """"Solver for the k-median problem that uses the iterative mistake minimization method, that solves the
    k-median problem with an explainable solution."""
    @staticmethod
    def name():
        return "IMM"
    def __call__(self, instance, pre_clusters):
        """
        Solve a k-median problem with the iterative mistake minimization algorithm
        @param instance: instance of the k-median problem
        @return: an explainable solution to the k-median problem
        """
        leaves, split_nodes = build_tree(instance, pre_clusters)

        return ExplainableOutput(instance, leaves, split_nodes, pre_clusters)


def build_tree(instance: Instance, pre_clusters: dict):
    dim = instance.dimension()
    leaves = []
    split_nodes = []
    root = ClusterNode(pre_clusters, np.array([[-inf, inf]] * dim))  # initial bounds are -inf, inf

    def rec_build_tree(node: ClusterNode):
        if node.is_homogeneous():
            leaves.append(node)
        else:
            i, theta, node_L, node_R = find_split(node)
            node.split = i, theta
            split_nodes.append(node)
            node.children = [node_L, node_R]
            rec_build_tree(node_L)
            rec_build_tree(node_R)

    rec_build_tree(root)
    return leaves, split_nodes


def find_split(node: ClusterNode) -> Tuple[int, float, ClusterNode, ClusterNode]:
    def count_mistakes(i, theta):
        centers = node.centers()
        clusters = node.clusters
        center_point_pairs = [(center, point) for center in centers for point in clusters[center]]
        return sum(mistake(point, center, i, theta) for center, point in center_point_pairs)

    def find_best_split_dim(i):
        centers = node.centers()
        clusters = node.clusters
        center_coords = [center.coordinates[i] for center in centers]
        l_i, r_i = min(center_coords), max(center_coords)

        # iterate over all potential thetas
        point_coords = [point.coordinates[i] for l in clusters.values() for point in l if
                        l_i <= point.coordinates[i] <= r_i]
        point_coords.sort()
        # take the midpoint of consecutive coordinates to avoid equality issues
        theta_candidates = [(a + b) / 2.0 for a, b in
                            zip(point_coords, point_coords[1:])]  # 2.0 to avoid integer division
        # TODO: (implement the efficient way of counting mistakes while iterating over thetas)
        brute_force_compute = [(count_mistakes(i,theta), theta) for theta in theta_candidates]
        min_mistakes, best_theta = min(brute_force_compute, key=lambda entry: entry[0])
        return min_mistakes, i, best_theta

    # compute best splits in each dimension
    dimension = len(node.bounds)
    split_candidates = [find_best_split_dim(i) for i in range(dimension)]
    _, i, theta = min(split_candidates, key=lambda entry: entry[0])
    node_L, node_R = make_kids(node, i, theta)
    return i, theta, node_L, node_R


def mistake(point: Point, center: Point, i, theta) -> bool:
    return (point.coordinates[i] <= theta) != (center.coordinates[i] <= theta)
