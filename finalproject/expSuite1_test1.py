from final_project_part1 import *
import timeit
import matplotlib.pyplot as plt


# Generate 10 random complete graphs, each with 150 nodes and maximum edge value of 50
testGraphList = []
for i in range(10):
    testGraphList.append(create_random_complete_graph(150, 50))

relaxationValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dijkstra_avgTimes = []
dijkstra_approx_avgTimes = []

# use source node of 0 for both dijkstra and dijkstra_approx
for element in relaxationValues:
    dijkstra_time = 0
    dijkstra_approx_time = 0

    for graph in testGraphList:
        dijkstra_time += timeit.timeit(lambda: dijkstra(graph, 0), number=1)
        dijkstra_approx_time += timeit.timeit(
            lambda: dijkstra_approx(graph, 0, element), number=1
        )

    # for each k value, compute average time across all graphs
    dijkstra_avgTimes.append(dijkstra_time / len(testGraphList))
    dijkstra_approx_avgTimes.append(dijkstra_approx_time / len(testGraphList))

"""
print(relaxationValues)
print("Dijkstra Execution Times: {}".format(dijkstra_avgTimes))
print("------------------------------------------------------")
print("Dijkstra Approx Execution Times: {}".format(dijkstra_approx_avgTimes))
"""

plt.plot(relaxationValues, dijkstra_avgTimes, label="Dijkstra", color="red")
plt.plot(
    relaxationValues, dijkstra_approx_avgTimes, label="Dijkstra Approx", color="blue"
)
plt.xlabel("Number of Relaxations")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time of Dijkstra and Dijkstra_Approx")
plt.legend()

plt.show()
