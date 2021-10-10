import algorithms.random_centers
import algorithms.kmedpluplus
import data.synthetic
import visualization.clusterings

instance = data.synthetic.k_clusters(5)
# solver = algorithms.random_centers.RandomCenters()
solver = algorithms.kmedpluplus.KMedPlusPlus()
output = solver(instance)
visualization.clusterings.show_clusters(output)

