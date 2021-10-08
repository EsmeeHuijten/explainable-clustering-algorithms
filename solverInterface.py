import abc
import numpy as np
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
        # pointlist = self.instance.points
        # pointarray = np.array(pointlist)
        # print(pointarray)
        # new_pointlist = [list(x) for x in pointarray]
        # for point in self.instance.points:
        #     [i in range(len(self.instance.points)) if pointarray[i, :] == point]
        return {i: closest_center(self.instance.points[i], self.centers) for i in
                range(len(self.instance.points))}
        # return {pointlist.index(np.array([point[0], point[1]])): closest_center(point, self.centers) for point in self.instance.points}

    def clusters(self):
        pointlist = self.instance.points
        return {i: [index for index, entry in self.assignment().items() if entry[0] == i] for i in
                range(self.instance.k)}

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
