import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def show_clusters(output):
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

    for i in range(len(output.centers)):
        x = [point.coordinates[0] for point in clusterpoints[output.centers[i]]]
        y = [point.coordinates[1] for point in clusterpoints[output.centers[i]]]
        plt.scatter(x, y, color=colors[i])
    plt.show()
