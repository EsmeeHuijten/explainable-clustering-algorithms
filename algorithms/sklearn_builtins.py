from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

from util import Point


@dataclass
class SKLearn:
    @staticmethod
    def name():
        return "SKLearn"

    def __call__(self, instance, _):
        # run KMeans
        kmeans_in = np.array([point.coordinates for point in instance.points])
        kmeans_out = KMeans(n_clusters=instance.k, random_state=0).fit(kmeans_in)

        # extract pre_clusters from KMeans
        pre_centers = [Point(centercoord) for centercoord in kmeans_out.cluster_centers_]
        pre_clusters = {center: [point for point, label in zip(kmeans_in, kmeans_out.labels_) if
                                 label == pre_centers.index(center)] for center in
                        pre_centers}

        # run DecisionTreeClassifier
        X = np.array(kmeans_in)
        Y = kmeans_out.labels_
        dec_tree = DecisionTreeClassifier(max_leaf_nodes=instance.k)
        dec_tree = dec_tree.fit(X, Y)
        plot_tree(dec_tree)
        plt.show()

        # TODO: build ExplainableOutput from dec_tree

        return None #ExplainableOutput(instance, leaves, split_nodes, pre_clusters)
