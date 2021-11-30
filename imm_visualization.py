import algorithms.iterative_mistake_minimization
import algorithms.Esfandiari_algorithm
import algorithms.Makarychev_algorithm
import data.mall.conversion
import visualization.clusterings
import visualization.imm_with_precluster
#TODO: separate visualization files for MakarychevAlgorithm and EsfandiariAlgorithm
mall_parameters = {'age': False, 'income': True, 'spending_score': True}
# instance = data.mall.conversion.get_instance(k=5, parameters=mall_parameters)
import data.synthetic
instance = data.synthetic.k_clusters(k=5, spread=1.2, cluster_size=50)
solver = algorithms.iterative_mistake_minimization.IMM()
# solver = algorithms.Esfandiari_algorithm.EsfandiariAlgorithm()
# solver = algorithms.Makarychev_algorithm.MakarychevAlgorithm()
output = solver(instance)
visualization.imm_with_precluster.show_explainable_clusters(output)
#visualization.clusterings.show_explainable_clusters(output)
