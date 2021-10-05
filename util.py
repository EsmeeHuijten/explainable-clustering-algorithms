import numpy as np


def dist(p, q, _norm=np.linalg.norm):  # putting np.linalg.norm as default argument avoids lookup of np at each call
    'the standard Euclidean norm'
    return _norm(p - q)


def closest_center(point, centers):
    dists = [dist(point, center) for center in centers]
    return np.argmin(dists), min(dists)
