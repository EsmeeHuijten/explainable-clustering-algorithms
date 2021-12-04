import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from solver_interface import CenterOutput, ExplainableOutput
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def show_explainable_clusters(output: ExplainableOutput):
    """
    Plots datapoints, showing the clusters with a colourcode
    @param output: an explainable solution to the k-median problem
    """
    k = len(output.medians)
    clusterpoints = output.clusters

    colors = cm.rainbow(np.linspace(0, 1, k))  # get a selection of evenly distributed colors

    xcoords = [point.coordinates[0] for point in output.instance.points]
    ycoords = [point.coordinates[1] for point in output.instance.points]
    xmin, xmax = min(xcoords), max(xcoords)
    ymin, ymax = min(ycoords), max(ycoords)

    fig, ax = plt.subplots()
    for i in range(k):
        x = [point.coordinates[0] for point in clusterpoints[output.medians[i]]]
        y = [point.coordinates[1] for point in clusterpoints[output.medians[i]]]
        plt.scatter(x, y, color=colors[i])
        bounds = output.leaves[i].bounds
        xminc, xmaxc, yminc, ymaxc = max(bounds[0][0], xmin), min(bounds[0][1], xmax), max(bounds[1][0], ymin), \
                                     min(bounds[1][1], ymax)
        box = [Rectangle((xminc, yminc), xmaxc-xminc, ymaxc-yminc, facecolor=colors[i])]
        pc = PatchCollection(box, facecolor = colors[i], alpha=0.4)
        ax.add_collection(pc)
        # plt.hlines(yminc, xminc, xmaxc, color='black')
        # plt.hlines(ymaxc, xminc, xmaxc, color='black')
        # plt.vlines(xminc, yminc, ymaxc, color='black')
        # plt.vlines(xmaxc, yminc, ymaxc, color='black')

    plt.show()

def show_clusters(output: CenterOutput):
    """
    Plots datapoints, showing the centers and the clusters with a colourcode
    @param output: a solution to the k-median problem
    """
    k = len(output.centers)
    clusterpoints = output.clusters()

    colors = cm.rainbow(np.linspace(0, 1, k))  # get a selection of evenly distributed colors

    xcoords = [point.coordinates[0] for point in output.instance.points]
    ycoords = [point.coordinates[1] for point in output.instance.points]
    xmin, xmax = min(xcoords), max(xcoords)
    ymin, ymax = min(ycoords), max(ycoords)

    plt.xlim(left=xmin - (xmax - xmin) * 0.1, right=xmax + (xmax - xmin) * 0.1)
    plt.ylim(bottom=ymin - (ymax - ymin) * 0.1, top=ymax + (ymax - ymin) * 0.1)
    plt.axis('off')

    xs = [center.coordinates[0] for center in output.centers]
    ys = [center.coordinates[1] for center in output.centers]
    plt.scatter(xs, ys, marker="o", s=150, color="black")

    for i in range(k):
        x = [point.coordinates[0] for point in clusterpoints[output.centers[i]]]
        y = [point.coordinates[1] for point in clusterpoints[output.centers[i]]]
        plt.scatter(x, y, color=colors[i])
    plt.show()
