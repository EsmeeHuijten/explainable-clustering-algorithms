import abc
from dataclasses import dataclass

import numpy as np

# TODO: what do we assume about our inputs? d-dimensional Eucl space? (then don't check for validity)

@dataclass
class Instance:
    points: list[np.ndarray]
    k: int


@dataclass
class Output:
    # TODO: how to implement Output? centers, clusters, decision tree? Separate class for explainable clusterings?
    instance: Instance
    centers: list
    clusters: list

    def cost(self):
        # TODO: implement this:
        raise NotImplementedError


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
