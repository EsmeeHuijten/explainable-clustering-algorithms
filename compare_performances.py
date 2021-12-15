import time

import matplotlib.pyplot as plt
import numpy as np
from data.creditcard.conversion_creditcard import get_instance as get_creditcard_instance
from algorithms import kmedplusplus, iterative_mistake_minimization, Makarychev_algorithm, Esfandiari_algorithm, \
    sklearn_builtins
from data.synthetic import k_clusters

k = 3
# do it again for k = 20 or k = 10 if it takes too long
creditcard_parameters = {'CUST_ID': False, 'BALANCE': True, 'BALANCE_FREQUENCY': False, 'PURCHASES': True,
                         'ONEOFF_PURCHASES': False, 'INSTALLMENTS_PURCHASES': False, 'CASH_ADVANCES': False,
                         'PURCHASES_FREQUENCY': False, 'ONEOFF_PURCHASES_FREQUENCY': False,
                         'PURCHASES_INSTALLMENTS_FREQUENCY': False, 'CASH_ADVANCE_FREQUENCY': False,
                         'CASH_ADVANCE_TRX': False, 'PURCHASES_TRX': False, 'CREDIT_LIMIT': False, 'PAYMENTS': False,
                         'MINIMUM_PAYMENTS': False, 'PRC_FULL_PAYMENT': False, 'TENURE': False}
# instance = get_creditcard_instance(k=k, parameters=creditcard_parameters)
# create list of instances of size 2^k
instance_sizes = [2 ** t for t in range(3, 9)]  # 2 ** t has to be >= k for all t !
# instances = [k_clusters(k, cluster_size=round(size / float(k))) for size in instance_sizes]
instances = [get_creditcard_instance(k=k, parameters=creditcard_parameters, num_points = size) for size in instance_sizes]

# compute pre-clustering for each instance
pre_solver = kmedplusplus.KMedPlusPlus(numiter=15)
pre_solutions = [pre_solver(instance) for instance in instances]  # output type pre_solver(): CenterOutput
pre_clusters = [pre_solution.clusters() for pre_solution in pre_solutions]
pre_costs = [pre_solution.cost for pre_solution in pre_solutions]
setup = [(instances[i], pre_clusters[i]) for i in range(len(instances))]


def measure_performance(algo, instance, pre_clusters):
    algo_start = time.perf_counter()
    output = algo(instance, pre_clusters)
    cost = output.cost()
    algo_end = time.perf_counter()
    return algo_end - algo_start, cost


algoList = [# iterative_mistake_minimization.IMM(),
            # Makarychev_algorithm.MakarychevAlgorithm(),
            # Esfandiari_algorithm.EsfandiariAlgorithm(),
            sklearn_builtins.SKLearn()]

performances_dict = {type(algo).__name__: [tuple(measure_performance(algo, instance, pre_cluster))
                                           for instance, pre_cluster in setup] for algo in algoList}

figr, axr = plt.subplots()
axr.set_title("Running times of different algorithms")
#axr.set_yscale('log')

figc, axc = plt.subplots()
axc.set_title("Boxplots of costs for different algorithms")
x = instance_sizes
costs_norm = []
for algo in algoList:
    performances = performances_dict[type(algo).__name__]
    runtimes = [performance[0] for performance in performances]
    costs = [performance[1] for performance in performances]
    costs_norm.append(np.array(costs) / np.array(pre_costs))
    axr.plot(x, runtimes, label=type(algo).__name__)

axc.boxplot(costs_norm, labels=[type(algo).__name__ for algo in algoList])

axr.legend()
# axc.legend()
figr.savefig("Runtime.png")
figc.savefig("Cost.png")
