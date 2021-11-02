import algorithms.kmedplusplus
import data.mall.conversion
import visualization.clusterings

mall_parameters = {'age': False, 'income': True, 'spending_score': True}
instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
# TODO: replace by imm algorithm
solver = algorithms.kmedplusplus.KMedPlusPlus(numiter=5)
output = solver(instance)
# TODO: replace by show_explainable_clusters
visualization.clusterings.show_clusters(output)
