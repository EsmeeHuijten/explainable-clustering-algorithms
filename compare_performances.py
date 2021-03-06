import time

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from util import Point, dist, median_coordinatewise
from data.creditcard.conversion_creditcard import get_instance as get_creditcard_instance
from algorithms import kmedplusplus, iterative_mistake_minimization, Makarychev_algorithm, Esfandiari_algorithm, \
    sklearn_builtins
from data.synthetic import k_clusters


def cost(clusters) -> float:
    mediandict = {center: median_coordinatewise(clusters[center]) for center in clusters.keys()}
    return sum(  sum(dist(mediandict[center], point) for point in clusters[center]) for center in clusters.keys())
    #cost_tot = sum([]) for center in list(clusters.keys())])
    #return cost_tot

k = 5
# do it again for k = 20 or k = 10 if it takes too long
creditcard_parameters = {'CUST_ID': False, 'BALANCE': True, 'BALANCE_FREQUENCY': False, 'PURCHASES': True,
                         'ONEOFF_PURCHASES': False, 'INSTALLMENTS_PURCHASES': False, 'CASH_ADVANCES': False,
                         'PURCHASES_FREQUENCY': False, 'ONEOFF_PURCHASES_FREQUENCY': False,
                         'PURCHASES_INSTALLMENTS_FREQUENCY': False, 'CASH_ADVANCE_FREQUENCY': False,
                         'CASH_ADVANCE_TRX': False, 'PURCHASES_TRX': False, 'CREDIT_LIMIT': False, 'PAYMENTS': False,
                         'MINIMUM_PAYMENTS': False, 'PRC_FULL_PAYMENT': False, 'TENURE': False}
# instance = get_creditcard_instance(k=k, parameters=creditcard_parameters)
# create list of instances of size 2^k

imm_until = 5
instance_sizes = [2 ** t for t in range(4, 14)]  # 2 ** t has to be >= k for all t !
algos_until = len(instance_sizes)

# imm_until = 7
# algos_until = 17
# instance_sizes = [500 * t for t in range(1, algos_until)] #500
add_string = "_k=5_kmed_exp_limitsy_"

# instances = [k_clusters(k, cluster_size=round(size / float(k))) for size in instance_sizes]
instances = [get_creditcard_instance(k=k, parameters=creditcard_parameters, num_points=size) for size in instance_sizes]

# compute pre-clustering for each instance

# k-med++
pre_solver = kmedplusplus.KMedPlusPlus(numiter=7)
print("after pre-solver")
pre_solutions = [pre_solver(instance) for instance in instances]  # output type pre_solver(): CenterOutput
print("after pre-solutions")
pre_clusters = [pre_solution.clusters() for pre_solution in pre_solutions]
print("after pre-clusters")
pre_costs = [pre_solution.cost for pre_solution in pre_solutions]
print("after pre-costs")

# SKLearn k-means
# kmeans_in = [np.array([point.coordinates for point in instance.points]) for instance in instances]
# kmeans_out = [KMeans(n_clusters=instances[i].k, random_state=0).fit(kmeans_in[i]) for i in range(len(instances))]
# pre_clusters = []
# for i in range(len(instances)):
#     pre_centers = [Point(centercoord) for centercoord in kmeans_out[i].cluster_centers_]
#     #print(instances[i])
#     #print(kmeans_out[i])
#     pre_clusters.append({center: [point for point, label in zip(instances[i].points, kmeans_out[i].labels_) if
#                                 label == pre_centers.index(center)] for center in
#                        pre_centers})
# pre_costs = [cost(pre_cluster) for pre_cluster in pre_clusters]

setup = [(instances[i], pre_clusters[i]) for i in range(len(instances))]


def measure_performance(algo, instance, pre_clusters):
    print(f"Running {type(algo).__name__} on instance of size {len(instance.points)}")
    algo_start = time.perf_counter()
    # print("pre_clusters", type(pre_clusters))
    output = algo(instance, pre_clusters)
    cost_ = cost(output.clusters)
    algo_end = time.perf_counter()
    return algo_end - algo_start, cost_


algoList = [iterative_mistake_minimization.IMM(),
            Makarychev_algorithm.Makarychev(),
            Esfandiari_algorithm.Esfandiari(),
            sklearn_builtins.SKLearn()]

# performances_dict = {type(algo).__name__: [tuple(measure_performance(algo, instance, pre_cluster))
#                                            for instance, pre_cluster in setup] for algo in algoList}
num_instances_dict = {"IMM": imm_until, "Makarychev": algos_until, "Esfandiari": algos_until, "SKLearn": algos_until}

performances_dict = {type(algo).__name__: [tuple(measure_performance(algo, instance, pre_cluster))
                                           for instance, pre_cluster in setup[:num_instances_dict[type(algo).__name__]]]
                     for algo in algoList}

figr, axr = plt.subplots()
# axr.set_yscale('log')
axr.set_xlabel("Instance size")
axr.set_ylabel("Runtime (s)")
figc, axc = plt.subplots()
axc.set_ylabel("Normalized cost")
axc.set_ylim(0.94, 1.08)

x = instance_sizes
costs_norm = []

for algo in algoList:
    performances = performances_dict[type(algo).__name__]
    runtimes = [performance[0] for performance in performances]
    costs = [performance[1] for performance in performances]
    costs_norm.append(np.array(costs) / np.array(pre_costs[:num_instances_dict[type(algo).__name__]]))
    axr.plot(x[:num_instances_dict[type(algo).__name__]], runtimes, label=type(algo).__name__)

axc.boxplot(costs_norm, labels=[type(algo).__name__ for algo in algoList])

axr.legend()
figr.savefig("Runtime" + add_string + ".png")
figc.savefig("Cost" + add_string + ".png")


