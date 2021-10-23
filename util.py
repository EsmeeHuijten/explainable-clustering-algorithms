from __future__ import annotations # allows us to use Point in type hints of Point methods. this will be default in Python 3.10

from typing import Callable

import numpy as np


class Point:
    """represents a datapoint in a k-median problem instance"""
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def closest_center(self, centers: list[Point]) -> tuple[Point, np.float64]:
        """
        Computes the closest center from the point and the distance to it
        @param centers: list of centers to check
        @return: tuple containing the center and distance
        """
        dists = [dist(self, center) for center in centers]
        return centers[np.argmin(dists)], min(dists)


def dist(p: Point, q: Point,
         _norm: Callable[[Point,
                          Point], np.float64] = np.linalg.norm) -> np.float64:  # putting np.linalg.norm as default argument avoids lookup of np at each call
    """Calculates the standard Euclidean norm
    @param p, q: points to calculate Euclidean distance between
    @return: Euclidean distance between point p and point q
    """
    return _norm(p.coordinates - q.coordinates)
