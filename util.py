from typing import Callable, Any

import numpy as np


class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def closest_center(self, centers: list) -> tuple[Any, np.float64]:
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
    """the standard Euclidean norm"""
    return _norm(p.coordinates - q.coordinates)
