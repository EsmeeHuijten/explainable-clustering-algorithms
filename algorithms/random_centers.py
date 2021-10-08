import random

from solverInterface import Output, Instance


class RandomCenters:
    def __call__(self, instance: Instance) -> Output:
        centers = random.sample(instance.points, instance.k)
        return Output(instance, centers=centers)