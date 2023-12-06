from final_project_part1 import *
import timeit
import matplotlib.pyplot as plt


# Generate 10 random complete graphs, each with 50 nodes and maximum possible edge weight of 50
testGraphList = []
for i in range(10):
    testGraphList.append(create_random_complete_graph(50, 50))

relaxationValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bellman_ford_approx_avgTimes = []
dijkstra_approx_avgTimes = []

# use source node of 0 for both bellman_ford_approx and dijkstra_approx
for element in relaxationValues:
    bellman_ford_approx_time = 0
    dijkstra_approx_time = 0

    for graph in testGraphList:
        bellman_ford_approx_time += timeit.timeit(
            lambda: bellman_ford_approx(graph, 0, element), number=1
        )
        dijkstra_approx_time += timeit.timeit(
            lambda: dijkstra_approx(graph, 0, element), number=1
        )

    # for each k value, compute average time across all graphs
    bellman_ford_approx_avgTimes.append(bellman_ford_approx_time / len(testGraphList))
    dijkstra_approx_avgTimes.append(dijkstra_approx_time / len(testGraphList))

"""
print(relaxationValues)
print("Bellman Approx Execution Times: {}".format(bellman_ford_approx_avgTimes))
print("------------------------------------------------------")
print("Dijkstra Approx Execution Times: {}".format(dijkstra_approx_avgTimes))
"""


plt.plot(
    relaxationValues,
    bellman_ford_approx_avgTimes,
    label="Bellman Ford Approx",
    color="red",
)
plt.plot(
    relaxationValues, dijkstra_approx_avgTimes, label="Dijkstra Approx", color="blue"
)
plt.xlabel("Number of Relaxations")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time of Bellman Ford Approx and Dijkstra Approx")
plt.legend()

plt.show()
