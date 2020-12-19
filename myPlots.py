import main
import alg_cluster
import random
import matplotlib.pyplot as plt
import time

def get_random_clusters(num_clusters):
    result_list = []
    for num in range(num_clusters):
        result_list.append(alg_cluster.Cluster(set([num]), random.random()*2 - 1, random.random()*2 - 1,0,0))
    return result_list

x_data = []
y_data = []
x1_data = []
y1_data = []
for n in range(2,201):
    cluster_list = get_random_clusters(n)

    start_time = time.time()
    main.slow_closest_pair(cluster_list)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    x_data.append(n)
    y_data.append(elapsed_time)

    start_time = time.time()
    main.fast_closest_pair(cluster_list)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    x1_data.append(n)
    y1_data.append(elapsed_time)


plt.plot(x_data,y_data, "g-", label = "Slow Closest Pair")
plt.plot(x1_data,y1_data, "r-", label = "Fast Closest Pair")
plt.legend()
plt.title("Running Time Comparison of Closest Pair Algorithms")
plt.suptitle("Desktop Python (PyCharm)")
plt.xlabel("Number of Initial Clusters")
plt.ylabel("Running time (seconds)")
plt.savefig("performanceTime.png")
plt.show()
