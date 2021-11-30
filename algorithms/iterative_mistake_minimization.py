from __future__ import \
    annotations  # allows us to use ClusterNode in type hints, which will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf

import numpy as np

import algorithms.kmedplusplus
from solver_interface import Instance, ExplainableOutput, ClusterNode

@dataclass
class IMM:
    """"Solver for the k-median problem that uses the iterative mistake minimization method, that solves the
    k-median problem with an explainable solution."""
    @staticmethod
    def name():
        return "IMM"
    # TODO: refactor to __call__(self, instance, preclusters)
    def __call__(self, instance):
        """
        Solve a k-median problem with the iterative mistake minimization algorithm
        @param instance: instance of the k-median problem
        @return: an explainable solution to the k-median problem
        """
        leaves, split_nodes, preclusters = build_tree(instance)

        return ExplainableOutput(instance, leaves, split_nodes, preclusters)


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
    return leaves, split_nodes, pre_clusters
