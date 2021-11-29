from __future__ import \
    annotations  # allows us to use ClusterNode in type hints, which will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf

import numpy as np

import algorithms.kmedplusplus
from solver_interface import Instance, ExplainableOutput, ClusterNode, make_kids


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

        return ExplainableOutput(instance, leaves, split_nodes, preclusters)


def build_tree(instance: Instance, pre_solver=algorithms.kmedplusplus.KMedPlusPlus(numiter=5)):
    dim = instance.dimension()
    X = instance.points
    pre_solution = pre_solver(instance)
    u0 = ClusterNode(pre_solution.clusters(), np.array([[-inf, inf]] * dim), pre_solution.centers)  # root
    leaves = []
    split_nodes = []

    def median_split(u: ClusterNode):
        if u.is_homogeneous():
            leaves.append(u)
        else:
            split_nodes.append(u)
            centers = u.set
            a = [min([center.coordinates[i] for center in centers]) for i in range(dim)]
            b = [max([center.coordinates[i] for center in centers]) for i in range(dim)]
            R = [a[i] - b[i] for i in range(dim)]
            probabilities = [R[i] / sum(R) for i in range(dim)]
            r = np.random.choice(np.arange(0, dim),
                                 p=probabilities)  # np.random method needed for p argument!
            z = np.random.uniform(a[r], b[r])

            node_L, node_R = make_kids(u, r, z, True)
            u.children = [node_L, node_R]
            median_split(node_L)
            median_split(node_R)

    if not u0.is_homogeneous():
        median_split(u0)
    return leaves, split_nodes, pre_solution.clusters()

