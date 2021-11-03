from dataclasses import dataclass, field
from anytree import Node, RenderTree

import numpy as np

from util import Point


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

@dataclass
class DecisionTree:
    """
    class representing decision/threshold trees
    each element of the list cluster_bounds is an array with lower bound/upper bound pairs for each coordinate
    """
    cluster_bounds: list[np.ndarray]
    # root: Node
    # nodes: list[Node]



@dataclass
class ExplainableOutput:
    instance: Instance
    decision_tree: DecisionTree

    # TODO: after algorithm returns tree, compute medoids and return clusters as dict[Point, list[Point]] as output type
    def clusters(self) -> dict[int, list[Point]]:
        num_clusters_x = len(self.decision_tree.cluster_bounds[0])
        num_clusters_y = len(self.decision_tree.cluster_bounds[1])
        bounds = self.decision_tree.cluster_bounds
        return {i+j*num_clusters_x: [point for point in self.instance.points if ( (bounds[0][i][0] <= point.coordinates[0] <= bounds[0][i][1]) \
                and (bounds[1][j][0] <= point.coordinates <= bounds[1][j][1]) ) ] for i in range(num_clusters_x) for j in range(num_clusters_y)}
