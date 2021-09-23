import random

from solverInterface import Solver, Output, Instance


class RandomCenters(Solver):
    def solve(self, instance: Instance) -> Output:
        centers = random.sample(instance.points, instance.k)
        return Output(instance, centers=centers, clusters=[])
