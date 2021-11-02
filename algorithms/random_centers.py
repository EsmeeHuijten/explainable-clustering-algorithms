import random

from solver_interface import CenterOutput, Instance


class RandomCenters:
    def __call__(self, instance: Instance) -> CenterOutput:
        """
        Solve a k-median problem by assigning k random centers
        @param instance: instance of the k-median problem
        @return: a solution to the k-median problem
        """
        centers = random.sample(instance.points, instance.k)
        return CenterOutput(instance, centers=centers)
