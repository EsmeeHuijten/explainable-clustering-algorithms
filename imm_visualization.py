import algorithms.iterative_mistake_minimization
import data.mall.conversion
import visualization.clusterings

mall_parameters = {'age': False, 'income': True, 'spending_score': True}
instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
#import data.synthetic
#instance = data.synthetic.k_clusters(k=2, cluster_size=5)
solver = algorithms.iterative_mistake_minimization.IMM()
output = solver(instance)
# TODO: replace by show_explainable_clusters
visualization.clusterings.show_explainable_clusters(output)
