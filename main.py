import algorithms.randomized
import data.synthetic
import visualization.clusterings

instance = data.synthetic.k_clusters(5)
solver = algorithms.randomized.RandomCenters()
output = solver.solve(instance)
visualization.clusterings.show_clusters(output.instance.points, output.centers)
