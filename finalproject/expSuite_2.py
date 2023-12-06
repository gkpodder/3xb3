# SAME AS EXPERIMENT1 NEED TO RETHINK experiment suite experiment 1
from final_project_part1 import *
import timeit
import matplotlib.pyplot as plt

# step 1: Generate 1 complete graph with 30 nodes
# and each node can have max weight of 10


testGraph = create_random_complete_graph(200, 10)
# 10,15,20,25
relaxationValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dijkstra_AlgTime = []
dijkstraApprox_AlgTime = []

# step 2: measure the execution time of dijkstra and dijkstra_approx
# use source node of 0
for element in relaxationValues:
    dijkstra_AlgTime.append(timeit.timeit(lambda: dijkstra(testGraph, 0), number=25))
    dijkstraApprox_AlgTime.append(
        timeit.timeit(lambda: dijkstra_approx(testGraph, 0, element), number=25)
    )

print(relaxationValues)
print("Dijkstra Execution Times: {}".format(dijkstra_AlgTime))
print("Dijkstra Approx Execution Times: {}".format(dijkstraApprox_AlgTime))

plt.plot(relaxationValues, dijkstra_AlgTime, label="Dijkstra", color="red")
plt.plot(
    relaxationValues, dijkstraApprox_AlgTime, label="Dijkstra Approx", color="blue"
)
plt.xlabel("Num Relaxations")
plt.ylabel("Execution Times (seconds)")
plt.title("Dijkstra vs Dijkstra_Approx")
plt.legend()

plt.show()
