"""
This file corresponds to the first graded lab of 2XC3.
Feel free to modify and/or add functions to this file.
"""
import random
import timeit
import matplotlib.pyplot as plt

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]


# Creates a near sorted list by creating a random list, sorting it, then doing a random number of swaps
def create_near_sorted_list(length, max_value, swaps):
    L = create_random_list(length, max_value)
    L.sort()
    for _ in range(swaps):
        r1 = random.randint(0, length - 1)
        r2 = random.randint(0, length - 1)
        swap(L, r1, r2)
    return L


# I have created this function to make the sorting algorithm code read easier
def swap(L, i, j):
    L[i], L[j] = L[j], L[i]


# ******************* Insertion sort code *******************

# This is the traditional implementation of Insertion Sort.
def insertion_sort(L):
    for i in range(1, len(L)):
        insert(L, i)


def insert(L, i):
    while i > 0:
        if L[i] < L[i-1]:
            swap(L, i-1, i)
            i -= 1
        else:
            return

# This is the variation of Insertion Sort
def insertion_sort2(L):
  for i in range(1,len(L)):
    value = L[i]
    while i > 0:
      if value < L[i-1]:
        L[i] = L[i-1]
      else:
        L[i] = value
        break
      i = i - 1

    if i == 0:
      L[i] = value

  return L

#Part 1 Experiment 2 Insertion Sort v.s. Insertion Sort 2 Testing
testCasesList = [create_random_list(10,10),create_random_list(50,50),create_random_list(100,100),create_random_list(250,250),
             create_random_list(500,500),create_random_list(750,750),create_random_list(1000,1000),create_random_list(1250,1250),
             create_random_list(1500,1500),create_random_list(2000,2000),create_random_list(2500,2500),create_random_list(3500,3500)
             ]
testTimes = {"insertion_sort": [], "insertion_sort2": []}
testLengths = []

for testCase in testCasesList:
    insertionSortTime = timeit.timeit(lambda: insertion_sort(testCase),number = 10)
    insertionSort2Time = timeit.timeit(lambda: insertion_sort2(testCase),number = 10)
    testTimes["insertion_sort"].append(insertionSortTime)
    testTimes["insertion_sort2"].append(insertionSort2Time)
    testLengths.append(len(testCase))

print("Insertion Sort times: {}".format(testTimes["insertion_sort"]))
print("Insertion Sort 2 times: {}".format(testTimes["insertion_sort2"]))

plt.plot(testLengths, testTimes["insertion_sort"],label="insertion_sort", color = "red")
plt.plot(testLengths,testTimes["insertion_sort2"],label="insertion_sort2", color = "blue")
plt.xlabel("List Length")
plt.ylabel("Elapsed Time")
plt.title("Insertion Sort vs Modified Insertion Sort")
plt.legend()

plt.show()



# ******************* Bubble sort code *******************

# Traditional Bubble sort
def bubble_sort(L):
    for i in range(len(L)):
        for j in range(len(L) - 1):
            if L[j] > L[j+1]:
                swap(L, j, j+1)

#This is variation of Bubble Sort
def bubble_sort2(L):
    for i in range(len(L)):
        val = None

        for j in range(len(L) - 1 - i):
            if val is None:
                if L[j] > L[j+1]:
                    val = L[j]
                    L[j] = L[j+1]
            
            else:
                if val > L[j+1]:
                    L[j] = L[j+1]
                
                else:
                    L[j] = val
                    val = None
        
        if val is not None: 
            L[len(L) - 1 - i] = val

#Part 1 Experiment 2 Bubble Sort v.s. Bubble Sort 2 Testing
testCasesList = [create_random_list(10,10),create_random_list(50,50),create_random_list(100,100),create_random_list(250,250),
             create_random_list(500,500),create_random_list(750,750),create_random_list(1000,1000),create_random_list(1250,1250),
             create_random_list(1500,1500),create_random_list(2000,2000),create_random_list(2500,2500),create_random_list(3500,3500)
             ]
testTimes = {"bubble_sort": [], "bubble_sort2": []}
testLengths = []

for testCase in testCasesList:
    bubbleSortTime = timeit.timeit(lambda: bubble_sort(testCase),number = 10)
    bubbleSort2Time = timeit.timeit(lambda: bubble_sort2(testCase),number = 10)
    testTimes["bubble_sort"].append(bubbleSortTime)
    testTimes["bubble_sort2"].append(bubbleSort2Time)
    testLengths.append(len(testCase))

print("Bubble Sort times: {}".format(testTimes["bubble_sort"]))
print("Bubble Sort 2 times: {}".format(testTimes["bubble_sort2"]))

plt.plot(testLengths, testTimes["bubble_sort"],label="bubble_sort", color = "red")
plt.plot(testLengths,testTimes["bubble_sort2"],label="bubble_sort2", color = "blue")
plt.xlabel("List Length")
plt.ylabel("Elapsed Time")
plt.title("Bubble Sort vs Modified Bubble Sort")
plt.legend()

plt.show()
# ******************* Selection sort code *******************

# Traditional Selection sort
def selection_sort(L):
    for i in range(len(L)):
        min_index = find_min_index(L, i)
        swap(L, i, min_index)


def find_min_index(L, n):
    min_index = n
    for i in range(n+1, len(L)):
        if L[i] < L[min_index]:
            min_index = i
    return min_index

#tests
Test_cases = [create_random_list(10,10),create_random_list(50,50),create_random_list(100,100),create_random_list(250,250),create_random_list(500,500),create_random_list(750,750),create_random_list(1000,1000),create_random_list(1250,1250),create_random_list(1500,1500),create_random_list(2000,2000),create_random_list(2500,2500),create_random_list(3500,3500)]
times = {"insertion_sort":[],"bubble_sort":[],"selection_sort":[]}
list_length = []
for L in Test_cases:
    time1 = timeit.timeit(lambda:insertion_sort(L),number = 10)
    time2 = timeit.timeit(lambda:bubble_sort(L),number = 10)
    time3 = timeit.timeit(lambda:selection_sort(L),number = 10)
    times.get("insertion_sort").append(time1)
    times.get("bubble_sort").append(time2)
    times.get("selection_sort").append(time3)
    list_length.append(len(L))

#uncomment this print statement
#print(times["bubble_sort"])
# Extract sorting times into a list
sorting_times = [times["insertion_sort"], times["bubble_sort"], times["selection_sort"]]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot list length vs. sorting time for each algorithm
ax.plot(list_length, sorting_times[0], label="Insertion Sort", color="blue")
ax.plot(list_length, sorting_times[1], label="Bubble Sort", color="green")
ax.plot(list_length, sorting_times[2], label="Selection Sort", color="red")

# Set axis labels and a legend
ax.set_xlabel("List Length")
ax.set_ylabel("Time")
ax.set_title("Sorting Algo Time vs. List Length")
ax.legend()

#uncomment the plot display
# Show the plot
#plt.show()

#Experiment 3
length=500
max_val=800

testTimes = {"insertion_sort": [], "bubble_sort": [],"selection_sort": []}
for num_swaps in range(0,500,10):
    list_insert=create_near_sorted_list(length,max_val,num_swaps)
    insertion_sort_time=timeit.timeit(lambda:insertion_sort(list_insert),number=10)
    testTimes["insertion_sort"].append(insertion_sort_time)

    list_bubble=create_near_sorted_list(length,max_val,num_swaps)
    bubble_sort_time=timeit.timeit(lambda:bubble_sort(list_bubble),number=10)
    testTimes["bubble_sort"].append(bubble_sort_time)

    list_select=create_near_sorted_list(length,max_val,num_swaps)
    selection_sort_time=timeit.timeit(lambda:selection_sort(list_select),number=10)
    testTimes["selection_sort"].append(selection_sort_time)
    print("done")

x_values = list(range(0, 500, 10))

# Plotting data for each sorting algorithm with specific colors
plt.plot(x_values, testTimes["insertion_sort"], label='Insertion Sort', color='red')
plt.plot(x_values, testTimes["bubble_sort"], label='Bubble Sort', color='green')
plt.plot(x_values, testTimes["selection_sort"], label='Selection Sort', color='blue')

# Adding labels, title, and legend
plt.xlabel('Number of Swaps')
plt.ylabel('Elapsed time in seconds')
plt.title('Swaps vs Time')
plt.legend()

# Display the plot
plt.show()



