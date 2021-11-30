import time

# create list of instances of size 2^k
from algorithms import kmedplusplus, iterative_mistake_minimization
from data.synthetic import k_clusters

k = 5
# do it again for k = 20 or k = 10 if it takes too long


instance_sizes = [2 ** t for t in range(3, 5)]
instances = [k_clusters(k, cluster_size=round(size / float(k))) for size in instance_sizes]

# compute preclustering for each instance
pre_solver = kmedplusplus.KMedPlusPlus(numiter=5)
pre_clusters = [pre_solver(instance) for instance in instances]
setup = zip(instances, pre_clusters)


# run each algorithm
# TODO: also keep track of cost
def measure_runtime(algo, instance, pre_clusters):
    algo_start = time.perf_counter()
    algo(instance, pre_clusters)
    algo_end = time.perf_counter()
    return algo_end - algo_start


algoList = [iterative_mistake_minimization.IMM, ]

run_times = {algo.name(): [measure_runtime(algo, instance, pre_cluster) for instance, pre_cluster in setup] for algo in
             algoList}

print(run_times)
# plot run times for each algo

# TODO: plot on log scale
# x = instance_sizes
# for algo in algoList:
#     y = run_times[algo.name()]
#     # plt.plot(x,y)
#     # plt.legend

# TODO: create box plot of price of explainability for each algo:
# compute ratio of cost of algo clustering divided by cost of pre_clustering
# for each algo, show box with all of its ratios
