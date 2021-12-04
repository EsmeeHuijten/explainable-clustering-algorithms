import time
import numpy as np
from algorithms import kmedplusplus, iterative_mistake_minimization, Makarychev_algorithm, Esfandiari_algorithm
from data.synthetic import k_clusters
import matplotlib.pyplot as plt

k = 5
# do it again for k = 20 or k = 10 if it takes too long

# create list of instances of size 2^k
instance_sizes = [2 ** t for t in range(3, 7)] #(3, 5)
instances = [k_clusters(k, cluster_size=round(size / float(k))) for size in instance_sizes]

# compute pre-clustering for each instance
pre_solver = kmedplusplus.KMedPlusPlus(numiter=15)
pre_solutions = [pre_solver(instance) for instance in instances]  #output type pre_solver(): CenterOutput
pre_clusters = [pre_solution.clusters() for pre_solution in pre_solutions]
pre_costs = [pre_solution.cost for pre_solution in pre_solutions]
setup = [(instances[i], pre_clusters[i]) for i in range(len(instances))]
# setup = zip(instances, pre_clusters)


# run each algorithm
def measure_performance(algo, instance, pre_clusters):
    algo_start = time.perf_counter()
    output = algo(instance, pre_clusters)
    # if algo ==iterative_mistake_minimization.IMM():
    #     print("hello")
    #     print("output", output)

    cost = output.cost()
    algo_end = time.perf_counter()
    return algo_end - algo_start, cost


algoList = [iterative_mistake_minimization.IMM(), Makarychev_algorithm.MakarychevAlgorithm(),
            Esfandiari_algorithm.EsfandiariAlgorithm()]

performances_dict = {algo.name(): [tuple(measure_performance(algo, instance, pre_cluster))
                              for instance, pre_cluster in setup] for algo in algoList}

figr, axr = plt.subplots()
axr.set_title("Running times of different algorithms")
axr.set_yscale('log')

figc, axc = plt.subplots()
axc.set_title("Boxplots of costs for different algorithms")
x = instance_sizes
costs_norm = []
for algo in algoList:
    performances = performances_dict[algo.name()]
    runtimes = [performance[0] for performance in performances]
    costs = [performance[1] for performance in performances]
    costs_norm.append(np.array(costs) / np.array(pre_costs))
    axr.plot(x, runtimes, label=algo.name())

axc.boxplot(costs_norm, labels=[algo.name() for algo in algoList])

axr.legend()
axc.legend()
figr.savefig("Runtime.png")
figc.savefig("Cost.png")
# compute ratio of cost of algo clustering divided by cost of pre_clustering
# for each algo, show box with all of its ratios
