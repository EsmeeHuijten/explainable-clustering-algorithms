import algorithms.kmedpluplus
import data.synthetic
import visualization.clusterings

instance = data.synthetic.k_clusters(k=5, cluster_size=10, spread=1.5)
# solver = algorithms.random_centers.RandomCenters()
solver = algorithms.kmedpluplus.KMedPlusPlus(numiter=5)
output = solver(instance)
visualization.clusterings.show_clusters(output)
