from dataclasses import dataclass, field

from util import Point


@dataclass
class Instance:
    """represents an instance of the k-median problem
    assume all instances are valid d-dimensional euclidean instances"""
    points: list[Point]  # each point is np.array
    k: int  # number of centers to be opened


@dataclass
class Output:
    # TODO: how to implement Output? centers, clusters, decision tree? Separate class for explainable clusterings?
    instance: Instance
    centers: list[Point]
    assignment: dict[Point, Point] = field(init=False)
    cost: float = field(init=False)

    def __post_init__(self):
        self.assignment = {point: point.closest_center(self.centers) for point in self.instance.points}
        self.cost = sum([self.assignment[point][1] for point in self.instance.points])

    def clusters(self) -> dict[Point, list[Point]]:
        return {center: [point for point in self.instance.points if self.assignment[point][0] == center] for center in
                self.centers}


class DecisionTree:
    pass


@dataclass
class ExplainableOutput(Output):
    decision_tree: DecisionTree
