from __future__ import \
    annotations  # allows us to use ClusterNode in type hints of ClusterNode methods. this will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf

import numpy as np

import algorithms.kmedplusplus
from solver_interface import Point, Instance, CenterOutput, ExplainableOutput, ClusterNode

@dataclass
class EsfandiariAlgorithm:
    """"Solver for the k-median problem that uses the method proposed by Makarychev et al, that solves the
    k-median problem with an explainable solution."""
    def __call__(self, instance):
        """
        Solve a k-median problem with the method proposed by Makarychev et al
        @param instance: instance of the k-median problem
        @return: an explainable solution to the k-median problem
        """
        leaves, split_nodes, preclusters = build_tree(instance)

        # TODO: let this actually return ExplainableOutput. it currently
        return ExplainableOutput(instance, leaves, split_nodes), preclusters

def build_tree(instance: Instance, pre_solver=algorithms.kmedplusplus.KMedPlusPlus(numiter=5)):
    dim = instance.dimension()
    X = instance.points
    pre_solution = pre_solver(instance)
    u0 = ClusterNode(pre_solution.clusters(), np.array([[-inf, inf]] * dim)) #root
    leaves = []
    split_nodes = []
    def median_split(u: ClusterNode):
        if u.is_homogeneous():
            leaves.append(u)
        else:
            split_nodes.append(u)
            centers = u.centers()
            dim = len(centers[0].coordinates)
            a = [min([center.coordinates[i] for center in centers]) for i in range(dim)]
            b = [max([center.coordinates[i] for center in centers]) for i in range(dim)]
            R = [a[i] - b[i] for i in range(dim)]
            probs = [R[i]/sum(R) for i in range(dim)]
            r = np.random.choice(np.arange(0, dim),
                                         p=probs)  # np.random method needed for p argument!
            z = np.random.uniform(a[r], b[r])
            node_L_centers = [center for center in centers if center.coordinates[r] <= z]
            node_L_clusters = {center: [point for point in u.clusters[center] if point.coordinates[r] <= z] for
                               center in node_L_centers}
            node_L_bounds = u.bounds.copy()
            node_L_bounds[r][1] = z # change upper bound to theta
            node_L = ClusterNode(node_L_clusters, node_L_bounds)

            node_R_centers = [center for center in centers if center.coordinates[r] > z]
            node_R_clusters = {center: [point for point in u.clusters[center] if point.coordinates[r] > z] for
                               center in node_R_centers}
            node_R_bounds = u.bounds.copy()
            node_R_bounds[r][0] = z  # change lower bound to theta
            node_R = ClusterNode(node_R_clusters, node_R_bounds)

            u.children = [node_L, node_R]
            median_split(node_L)
            median_split(node_R)

    while multiple_center_leaf(u0):
        median_split(u0)
    return 0

def multiple_center_leaf(u: ClusterNode):
    if u.children:
        return max(multiple_center_leaf(u.children[0]), multiple_center_leaf(u.children[1]))
    return (len(u.set) >= 2)


def build_tree(instance: Instance, pre_solver=algorithms.kmedplusplus.KMedPlusPlus(numiter=5)):
    dim = instance.dimension()
    leaves = []
    split_nodes = []
    pre_clusters = pre_solver(instance).clusters()
    root = ClusterNode(pre_clusters, np.array([[-inf, inf]] * dim))  # initial bounds are -inf, inf

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
    return leaves, split_nodes, pre_clusters
