from dataclasses import dataclass
from typing import Optional
from __future__ import annotations # allows us to use ClusterNode in type hints of ClusterNode methods. this will be default in Python 3.10


import algorithms
from solver_interface import Point, Instance, ExplainableOutput, DecisionTree, CenterOutput
import kmedplusplus
import numpy as np

@dataclass
class IMM:
    def __call__(self, instance) -> ExplainableOutput:
        return pass



@dataclass
class ClusterNode:
    clusters: dict[Point, list[Point]]
    split: Optional[(int, float)] = None
    children: list[ClusterNode] = []
    def is_homogeneous(self):
        return len(self.clusters.keys())==1
    # TODO: implement this
    def find_split(self):
        return pass


def build_tree(instance):
    leaves = []

    clusters = algorithms.kmedplusplus.KMedPlusPlus(instance).clusters()
    root = ClusterNode(clusters)
    rec_build_tree(root)
    # TODO: figure out what this should return, adjust ExplainableOutput class if required
    return leaves



def mistake(point: Point, center: Point, i, theta) -> bool:
    return (point.coordinates[i] <= theta) != (center.coordinates[i] <= theta)


def rec_build_tree(node: ClusterNode):
    if node.is_homogeneous():
        global leaves
        leaves.append(node)
        return node
    else:
        i, theta, node_L, node_R = node.find_split()
        node.split = i, theta
        node.children = [node_L, node_R]
        rec_build_tree(node_L)
        rec_build_tree(node_R)






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

