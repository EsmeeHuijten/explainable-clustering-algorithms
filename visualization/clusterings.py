from math import sqrt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# TODO: might want to put this somewhere else
def eucl_dist(p, q):
    'the standard Euclidean norm'
    return sqrt(sum([(p_i - q_i) ** 2 for p_i, q_i in zip(p, q)]))


# TODO: this computes k-center cost. compute k-median or delete if not necessary
def clusters(points, centers):
    '''for each point compute the (index of the) closest center
    and the cost of the clustering'''

    def closest_center_distance(point):
        dists = [eucl_dist(point, center) for center in centers]
        return np.argmin(dists), min(dists)

    closest_centers_distances = [closest_center_distance(point) for point in points]
    closest_centers = [entry[0] for entry in closest_centers_distances]
    cost = max([entry[1] for entry in closest_centers_distances])
    return closest_centers, round(cost, 2)


# TODO: refactor to take Output as input
def show_clusters(points, centers):
    k = len(centers)
    closest_center, cost = clusters(points, centers)

    colors = cm.rainbow(np.linspace(0, 1, k))  # get a selection of evenly distributed colors

    xcoords = [point[0] for point in points]
    ycoords = [point[1] for point in points]
    xmin, xmax = min(xcoords), max(xcoords)
    ymin, ymax = min(ycoords), max(ycoords)

    plt.xlim(left=xmin - (xmax - xmin) * 0.1, right=xmax + (xmax - xmin) * 0.1)
    plt.ylim(bottom=ymin - (ymax - ymin) * 0.1, top=ymax + (ymax - ymin) * 0.1)
    plt.axis('off')

    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    plt.scatter(xs, ys, marker="o", s=150, color="black")

    for i in range(len(points)):
        plt.scatter(points[i][0], points[i][1], color=colors[closest_center[i]])
    plt.show()
