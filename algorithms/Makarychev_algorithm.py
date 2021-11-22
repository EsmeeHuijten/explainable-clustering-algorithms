from __future__ import \
    annotations  # allows us to use ClusterNode in type hints of ClusterNode methods. this will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf

import numpy as np

import algorithms.kmedplusplus
from solver_interface import Point, Instance, CenterOutput, ExplainableOutput, ClusterNode

@dataclass
class MakarychevAlgorithm:
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
    centers = pre_solution.centers
    Xr = X + centers
    t = 0
    root = ClusterNode(pre_solution.clusters(), np.array([[-inf, inf]] * dim)) #T0
    #if Tt contains leaf (if contains leaf) with more than 1 center, do....

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
