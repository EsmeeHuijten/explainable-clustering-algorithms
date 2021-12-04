from __future__ import \
    annotations  # allows us to use ClusterNode in type hints, which will be default in Python 3.10

from dataclasses import dataclass, field
from math import inf
from util import dist
import numpy as np
from numpy import ndarray

import algorithms.kmedplusplus
from solver_interface import Instance, ExplainableOutput, ClusterNode, make_kids

@dataclass
class MakarychevAlgorithm:
    """"Solver for the k-median problem that uses the method proposed by Makarychev et al, that solves the
    k-median problem with an explainable solution."""
    @staticmethod
    def name():
        return "MakarychevAlgorithm"
    def __call__(self, instance, pre_clusters):
        """
        Solve a k-median problem with the method proposed by Makarychev et al
        @param instance: instance of the k-median problem
        @return: an explainable solution to the k-median problem
        """
        leaves, split_nodes = build_tree(instance, pre_clusters)

        return ExplainableOutput(instance, leaves, split_nodes, pre_clusters)


def build_tree(instance: Instance, pre_clusters: dict):
    dim = instance.dimension()
    X = instance.points
    leaves = []
    split_nodes = []
    centers = list(pre_clusters.keys())
    k = len(centers)
    Xr = X + centers
    T0 = ClusterNode(pre_clusters, np.array([[-inf, inf]] * dim), Xr) #root

    def rec_build_tree(T: ClusterNode):
        if T.is_homogeneous():
            leaves.append(T)
        else:
            centers = list(T.centers())
            S = [[[[min(center1.coordinates[0], center2.coordinates[0]),
                    max(center1.coordinates[0], center2.coordinates[0])],
                   [min(center1.coordinates[1], center2.coordinates[1]),
                    max(center1.coordinates[1], center2.coordinates[1])]] \
                  for center1 in centers] for center2 in centers]
            num_centers = len(centers)
            E = [(i, j) for i in range(num_centers) for j in range(num_centers) if (i != j) and (i < j)] #enough since tree is built top-down
            D = max([dist(centers[i], centers[j]) for (i, j) in E])
            R = [S[i][j] for (i, j) in E if mu(S[i][j]) > D/(k**3)]
            R_range = sum([mu(S) for S in R])
            R_cum = [sum([mu(R[j]) for j in range(i)]) for i in range(len(R)+1)]
            z = np.random.uniform(0.0, R_range)
            index_chosen = [i for i in range(len(R_cum)-1) if R_cum[i] <= z < R_cum[i+1]] #should be only 1!!
            S_chosen = R[index_chosen[0]]
            z_red = z - R_cum[index_chosen[0]]
            S_xrange = S_chosen[0][1] - S_chosen[0][0]
            if z_red <= S_xrange:
                i = 0
                theta = S_chosen[0][0] + z_red
            else:
                i = 1
                theta = S_chosen[1][0] + (z_red - S_xrange)
            split_nodes.append(T)
            node_L, node_R = make_kids(T, i, theta)
            rec_build_tree(node_L)
            rec_build_tree(node_R)

    rec_build_tree(T0)
    return leaves, split_nodes


def mu(S: ndarray):
    return (S[0][1]-S[0][0]) + (S[1][1] - S[1][0])


def nested_sum(L: list):
    return sum( nested_sum(x) if isinstance(x, list) else x for x in L )
