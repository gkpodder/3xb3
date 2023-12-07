from final_project_part1 import *
import math
import timeit
import matplotlib.pyplot as plt


# Generate x values to begin with
sampleInputList = [4, 8, 16, 32, 64, 128, 256, 512]
loggedInputList = []

# Generate random complete graphs with each element in sampleInputList representing number of nodes
testGraphList = []
for element in sampleInputList:
    testGraphList.append(create_random_complete_graph(element, 30))

mystery_TimesList = []
mystery_TimesList_Logged = []
# measure execution time that mystery_function takes on given graph
for graph in testGraphList:
    mysteryTime = 0
    mysteryTime += timeit.timeit(lambda: mystery(graph), number=1)
    mystery_TimesList.append(mysteryTime)

# apply log on each time in mystery_TimesList
for element in mystery_TimesList:
    mystery_TimesList_Logged.append(math.log2(element))


# Generate x,x^2,x^3 lines
x_linear = []
x_linear_logged = []

x_squared = []
x_squared_logged = []

x_cubed = []
x_cubed_logged = []

for element in sampleInputList:
    x_linear.append(element)
    x_squared.append(element * element)
    x_cubed.append(element * element * element)

# apply log on sampleInputList (x-log array)
for element in sampleInputList:
    loggedInputList.append(math.log2(element))

# apply log on x,x^2,x^3 line
for element in x_linear:
    x_linear_logged.append(math.log2(element))

for element in x_squared:
    x_squared_logged.append(math.log2(element))

for element in x_cubed:
    x_cubed_logged.append(math.log2(element))


# plot logged x-array, logged y-array
plt.plot(
    loggedInputList,
    mystery_TimesList_Logged,
    label="mystery function",
    color="black",
)
plt.plot(loggedInputList, x_linear_logged, label="x", color="red")
plt.plot(
    loggedInputList,
    x_squared_logged,
    label="x^2",
    color="green",
)
plt.plot(loggedInputList, x_cubed_logged, label="x^3", color="blue")
plt.xlabel("log (nodes)")
plt.ylabel("log (execution time) (seconds)")
plt.title("Mystery Function Time Complexity Analysis")
plt.legend()

plt.show()
