from final_project_part1 import *
import matplotlib.pyplot as plt


# Generate 10 random complete graphs, each with 150 nodes and maximum possible edge weight of 50
testGraphList = []
for i in range(10):
    testGraphList.append(create_random_complete_graph(150, 50))

relaxationValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dijkstra_avgDistances = []
dijkstra_approx_avgDistances = []

# use source node of 0 for both dijkstra and dijkstra_approx
for element in relaxationValues:
    dijkstra_dist = 0
    dijkstra_approx_dist = 0

    for graph in testGraphList:
        dijkstra_dist += total_dist(dijkstra(graph, 0))
        dijkstra_approx_dist += total_dist(dijkstra_approx(graph, 0, element))

    # for each k value, compute the average distances across the 10 graphs
    dijkstra_avgDistances.append(dijkstra_dist / len(testGraphList))
    dijkstra_approx_avgDistances.append(dijkstra_approx_dist / len(testGraphList))


plt.plot(relaxationValues, dijkstra_avgDistances, label="Dijkstra", color="red")
plt.plot(
    relaxationValues,
    dijkstra_approx_avgDistances,
    label="Dijkstra Approx",
    color="blue",
)
plt.xlabel("Number of Relaxations")
plt.ylabel("Average total distance")
plt.title("Average total distance of Dijkstra and Dijkstra_Approx")
plt.legend()

plt.show()
