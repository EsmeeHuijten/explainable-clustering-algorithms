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
         _norm: Callable[[Point], np.float64] = np.linalg.norm) -> np.float64:  # putting np.linalg.norm as default argument avoids lookup of np at each call
    """Calculates the standard Euclidean norm
    @param p, q: points to calculate Euclidean distance between
    @return: Euclidean distance between point p and point q
    """
    return _norm(p.coordinates - q.coordinates)


def closest_to_centroid(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster, this function computes the centroid of that cluster
    and returns the point closest to the centroid.
    @param clusterpoints: all points of cluster in question
    @return: the datapoint closest to the centroid of the cluster in question
    """
    cluster_points_x = [point.coordinates[0] for point in clusterpoints]
    cluster_points_y = [point.coordinates[1] for point in clusterpoints]
    centroid = Point((np.average(cluster_points_x), np.average(cluster_points_y)))
    dists = [dist(point, centroid) for point in clusterpoints]
    new_center = clusterpoints[np.argmin(dists)]
    return new_center


def medoid_bruteforce(clusterpoints: list[Point]) -> Point:
    """
    Given a cluster, this function computes the medoid of that cluster
    and returns it. This function calculates the medoid with the brute force method.
    @param clusterpoints: points of cluster in question
    @return: medoid of the cluster in question
    """
    cost = [sum([dist(point1, point2) for point2 in clusterpoints]) for point1 in clusterpoints]
    return clusterpoints[np.argmin(cost)]