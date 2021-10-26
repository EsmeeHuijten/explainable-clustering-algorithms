from dataclasses import dataclass, field

import numpy as np

from util import Point


@dataclass
class Instance:
    """represents an instance of the k-median problem
    assume all instances are valid d-dimensional euclidean instances"""
    points: list[Point]  # each point is np.array
    k: int  # number of centers to be opened


@dataclass
class Output:
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


@dataclass
class ExplainableOutput(Output):
    decision_tree: DecisionTree
