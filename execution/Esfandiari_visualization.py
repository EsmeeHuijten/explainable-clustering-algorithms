import sys
sys.path.append('..')
sys.path.append('../..')

import algorithms.Esfandiari_algorithm
import data.mall.conversion
import visualization.clusterings
import visualization.imm_with_precluster

mall_parameters = {'age': False, 'income': True, 'spending_score': True}
instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
# import data.synthetic
# instance = data.synthetic.k_clusters(k=5, spread=1.2, cluster_size=50)
pre_solver = algorithms.kmedplusplus.KMedPlusPlus(numiter=5)
pre_clusters = pre_solver(instance).clusters()
solver = algorithms.Esfandiari_algorithm.Esfandiari()
output = solver(instance, pre_clusters)
# visualization.imm_with_precluster.show_explainable_clusters(output)
visualization.clusterings.show_explainable_clusters(output)
