import abc
from dataclasses import dataclass

from util import closest_center


@dataclass
class Instance:
    """represents an instance of the k-median problem
    assume all instances are valid d-dimensional euclidean instances"""
    points: list  # each point is np.array
    k: int  # number of centers to be opened


@dataclass
class Output:
    # TODO: how to implement Output? centers, clusters, decision tree? Separate class for explainable clusterings?
    instance: Instance
    centers: list

    # TODO: implement this in a way such that assignment is only computed once
    def assignment(self):
        return {point: closest_center(point, self.centers) for point in self.instance.points}

    def clusters(self):
        return {i: [point for point,entry in self.assignment().items() if entry[0]==i] for i in range(self.instance.k)}

    def cost(self):
        return []


class DecisionTree:
    pass


@dataclass
class ExplainableOutput(Output):
    decision_tree: DecisionTree


# TODO: restricting type (hints) of implementation is tricky. so this does not do much...
class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self, instance: Instance) -> Output:
        pass
