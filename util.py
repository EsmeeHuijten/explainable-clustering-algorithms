import numpy as np


class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def closest_center(self, centers: list):
        dists = [dist(self, center) for center in centers]
        return centers[np.argmin(dists)], min(dists)


def dist(p, q, _norm=np.linalg.norm):  # putting np.linalg.norm as default argument avoids lookup of np at each call
    """the standard Euclidean norm"""
    return _norm(p.coordinates - q.coordinates)