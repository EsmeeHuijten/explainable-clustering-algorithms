from __future__ import \
    annotations  # allows us to use ClusterNode in type hints of ClusterNode methods. this will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf
from typing import Optional

import numpy as np
from numpy import ndarray

import algorithms.kmedplusplus
from solver_interface import Point, Instance, CenterOutput
from util import medoid_bruteforce


@dataclass
class IMM:
    def __call__(self, instance):
        leaves, split_nodes = build_tree(instance)

        # TODO: let this actually return ExplainableOutput. it currently
        centers = [medoid_bruteforce(list(node.clusters.values())[0]) for node in
                   leaves]  # for each leaf node, get list of points in (only) cluster
        return CenterOutput(instance, centers)


@dataclass
class ClusterNode:
    clusters: dict[Point, list[Point]]
    bounds: ndarray  # array of arrays (lower_bound, upper_bound) for each dimension
    split: Optional[(int, float)] = None
    children: list[ClusterNode] = field(default_factory=list)  # needed in order to get default value []

    def centers(self):
        return self.clusters.keys()

    def dimension(self):
        return len(list(self.clusters.keys())[0].coordinates)

    def is_homogeneous(self):
        return len(self.clusters.keys()) == 1

    def find_split(self) -> (int, float, ClusterNode, ClusterNode):
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


def build_tree(instance: Instance, pre_solver=algorithms.kmedplusplus.KMedPlusPlus(numiter=5)):
    dim = instance.dimension()
    leaves = []
    split_nodes = []
    clusters = pre_solver(instance).clusters()
    root = ClusterNode(clusters, np.array([[-inf, inf]] * dim))  # initial bounds are -inf, inf

    def rec_build_tree(node: ClusterNode):
        if node.is_homogeneous():
            leaves.append(node)
        else:
            i, theta, node_L, node_R = node.find_split()
            node.split = i, theta
            split_nodes.append(node)
            node.children = [node_L, node_R]
            rec_build_tree(node_L)
            rec_build_tree(node_R)

    rec_build_tree(root)
    # TODO: figure out what this should return, adjust ExplainableOutput class if required
    return leaves, split_nodes


def mistake(point: Point, center: Point, i, theta) -> bool:
    return (point.coordinates[i] <= theta) != (center.coordinates[i] <= theta)


'''


def build_tree(refset: CenterOutput, y: list[Point]) -> DecisionTree:
    cluster_bounds_final = [[], []]
    cluster_bounds, L, R = make_node(refset, y)
    while L != None:
        cluster_bounds_l, L_l, R_l = make_node(refset, L)
        cluster_bounds_r, L_r, R_r = make_node(refset, R)
        cluster_bounds_final[0] = cluster_bounds_final[0] + cluster_bounds[0]
        cluster_bounds_final[1] = cluster_bounds_final[1] + cluster_bounds[1]
    #TODO: call make_node again for each L and each R that comes out of make_node
    return 0

def make_node(refset: CenterOutput, y: list[Point]) -> list[float], list[Point], list[Point]:
    """"Looks for a tree-induced clustering that fits the labels y for this instance.
    @param refset: Our reference set of centers
    @param y:      List of closest center for each point
    @return:       Decision tree
    """
    x_coords = [point.coordinates[0] for point in refset.instance.points]
    y_coords = [point.coordinates[1] for point in refset.instance.points]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    #check for the case that the dataset is one big cluster
    if y.count(y[0]) == len(y):
        cluster_bounds = [[[x_min, x_max]], [[y_min, y_max]]]
        return cluster_bounds, None, None
    l = [min([point.coordinates[i] for point in y])for i in range(2)]
    r = [max([point.coordinates[i] for point in y])for i in range(2)]
    i_theta_list = [(i, point.coordinates[i]) for i in range(2) for point in refset.instance.points]
    index = np.argmin([sum([mistake(x, mu, i, point.coordinates[i]) for x in refset.instance.points for mu in list(set(y))]) \
             for i in range(2) for point in refset.instance.points if (l <= point.coordinates[i] <= r)])
    (i, theta) = i_theta_list[index]
    M = [point for point in refset.instance.points if (mistake(point, point.closest_center(refset.centers), i, theta) == 1)]
    L = [point for point in refset.instance.points if (point.coordinates[i] <= theta and (j not in M))]
    R = [point for point in refset.instance.points if (point.coordinates[i] > theta and (j not in M))]
    if i == 0:
        cluster_bounds = [[[x_min, theta], [theta, x_max]], [[y_min, y_max]]]
    else:
        cluster_bounds = [[[x_min, x_max]], [[y_min, theta], [theta, y_max]]]
    return cluster_bounds, L, R

class IMM:
    #TODO: docstring
    def __call__(self,instance: Instance) -> ExplainableOutput:
        # get a reference set of k centers, we now use kmedplusplus for this
        solver = kmedplusplus.KMedPlusPlus(numiter=5)
        refset = solver(instance)
        y = [point.closest_center(refset.centers)[0] for point in refset.instance.points] #list of centers
        output = build_tree(refset, y)
        return output

'''
