import algorithms.randomized
import algorithms.kmedpluplus
import data.synthetic
import visualization.clusterings

instance = data.synthetic.k_clusters(5)
# solver = algorithms.randomized.RandomCenters()
solver = algorithms.kmedpluplus.KMedian()
output = solver.solve(instance)
visualization.clusterings.show_clusters(output)

