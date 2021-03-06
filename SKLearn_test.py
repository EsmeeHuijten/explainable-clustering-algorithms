import algorithms.sklearn_builtins
import data.synthetic
import data.mall.conversion
import visualization.imm_with_precluster

instance = data.synthetic.k_clusters(k=5, cluster_size=10, spread=1.5)
#mall_parameters = {'age': False, 'income': True, 'spending_score': True}
#instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
solver = algorithms.sklearn_builtins.SKLearn()
output = solver(instance, None)
visualization.imm_with_precluster.show_explainable_clusters(output)