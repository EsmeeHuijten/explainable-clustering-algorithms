from dataclasses import dataclass
from math import inf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from solver_interface import ExplainableOutput, ClusterNode, make_kids
import matplotlib.pyplot as plt

from util import Point


@dataclass
class SKLearn:

    def __call__(self, instance, _):
        # run KMeans
        kmeans_in = np.array([point.coordinates for point in instance.points])
        kmeans_out = KMeans(n_clusters=instance.k, random_state=0).fit(kmeans_in)

        # extract pre_clusters from KMeans
        pre_centers = [Point(centercoord) for centercoord in kmeans_out.cluster_centers_]
        pre_clusters = {center: [point for point, label in zip(instance.points, kmeans_out.labels_) if
                                 label == pre_centers.index(center)] for center in
                        pre_centers}

        # run DecisionTreeClassifier
        X = np.array(kmeans_in)
        Y = kmeans_out.labels_
        dec_tree = DecisionTreeClassifier(max_leaf_nodes=instance.k)
        dec_tree = dec_tree.fit(X, Y)

        underlyingtree = dec_tree.tree_
        leaves = []
        split_nodes = []
        dim = instance.dimension()
        root = ClusterNode(pre_clusters, np.array([[-inf, inf]] * dim))

        def rec_tree(node: ClusterNode, j: int):
            i = underlyingtree.feature[j]
            theta = underlyingtree.threshold[j]

            if underlyingtree.children_left[j] == -1:
                leaves.append(node)
            else:
                split_nodes.append(node)
                node_L, node_R = make_kids(node, i, theta, False)
                rec_tree(node_L, underlyingtree.children_left[j])
                rec_tree(node_R, underlyingtree.children_right[j])
        rec_tree(root, 0)

        return ExplainableOutput(instance, leaves, split_nodes, pre_clusters)
