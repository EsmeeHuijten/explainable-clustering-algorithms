from dataclasses import dataclass
from math import inf
from sklearn.cluster import KMeans
from util import Point
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from solver_interface import ExplainableOutput, ClusterNode, make_kids


@dataclass
class SKLearn:

    def __call__(self, instance, pre_clusters):
        # run KMeans
        # kmeans_in = np.array([point.coordinates for point in instance.points])
        # kmeans_out = KMeans(n_clusters=instance.k, random_state=0).fit(kmeans_in)
        #
        # pre_centers = [Point(centercoord) for centercoord in kmeans_out.cluster_centers_]
        # pre_clusters = {center: [point for point, label in zip(instance.points, kmeans_out.labels_) if
        #                          label == pre_centers.index(center)] for center in
        #                 pre_centers}

        # run DecisionTreeClassifier
        instance_in = np.array([point.coordinates for point in instance.points])
        X = np.array(instance_in)

        def find_label(pre_clusters, point):
            labels = list(pre_clusters.keys())
            for center in pre_clusters.keys():
                if point in pre_clusters[center]:
                    return labels.index(center)

        Y = np.array([find_label(pre_clusters, point) for point in instance.points])
        #Y = kmeans_out.labels_
        dec_tree = DecisionTreeClassifier(max_leaf_nodes=instance.k)
        dec_tree = dec_tree.fit(X, Y)

        underlyingtree = dec_tree.tree_
        plot_tree(dec_tree)
        # plt.show()
        leaves = []
        split_nodes = []
        dim = instance.dimension()
        root = ClusterNode(pre_clusters, np.array([[-inf, inf]] * dim))

        def rec_tree(node: ClusterNode, j: int):
            if underlyingtree.children_left[j] == underlyingtree.children_right[j]:

                leaves.append(node)
            else:
                split_nodes.append(node)
                i = underlyingtree.feature[j]
                theta = underlyingtree.threshold[j]

                node_L, node_R = make_kids(node, i, theta, False)

                rec_tree(node_L, underlyingtree.children_left[j])
                rec_tree(node_R, underlyingtree.children_right[j])

        rec_tree(root, 0)

        return ExplainableOutput(instance, leaves, split_nodes, pre_clusters)
