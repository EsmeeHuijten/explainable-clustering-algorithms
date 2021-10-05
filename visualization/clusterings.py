import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from util import closest_center

def clusters(points, centers):
    '''for each point compute the (index of the) closest center
    and the cost of the clustering'''

    closest_centers_distances = [closest_center(point, centers) for point in points]
    closest_centers = [entry[0] for entry in closest_centers_distances]
    # cost = sum([entry[1] for entry in closest_centers_distances])
    return closest_centers


def show_clusters(output):
    k = len(output.centers)
    closest_center = clusters(output.instance.points, output.centers)

    colors = cm.rainbow(np.linspace(0, 1, k))  # get a selection of evenly distributed colors

    xcoords = [point[0] for point in output.instance.points]
    ycoords = [point[1] for point in output.instance.points]
    xmin, xmax = min(xcoords), max(xcoords)
    ymin, ymax = min(ycoords), max(ycoords)

    plt.xlim(left=xmin - (xmax - xmin) * 0.1, right=xmax + (xmax - xmin) * 0.1)
    plt.ylim(bottom=ymin - (ymax - ymin) * 0.1, top=ymax + (ymax - ymin) * 0.1)
    plt.axis('off')

    xs = [center[0] for center in output.centers]
    ys = [center[1] for center in output.centers]
    plt.scatter(xs, ys, marker="o", s=150, color="black")

    for i in range(len(output.instance.points)):
        plt.scatter(output.instance.points[i][0], output.instance.points[i][1], color=colors[closest_center[i]])
    plt.show()
