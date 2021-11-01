import algorithms.kmedplusplus
import data.synthetic
import data.mall.conversion
import visualization.clusterings

instance = data.synthetic.k_clusters(k=5, cluster_size=10, spread=1.5)
#mall_parameters = {'age': False, 'income': True, 'spending_score': True}
#instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
# solver = algorithms.random_centers.RandomCenters()
solver = algorithms.kmedplusplus.KMedPlusPlus(numiter=5)
output = solver(instance)
visualization.clusterings.show_clusters(output)
