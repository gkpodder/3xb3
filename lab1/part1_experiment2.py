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


# This is the variation of Selection Sort
#L is a list
def selection_sort2(L):
  low = 0
  high = len(L)-1
  while (low < high):
    minIndex = low
    maxIndex = high
    for i in range(low,high + 1):
      #In a given iteration, find min and max index
      if L[i] < L[minIndex]:
        minIndex = i

      if L[i] > L[maxIndex]:
        maxIndex = i

    #swapping logic
    minValue = L[minIndex]
    maxValue = L[maxIndex]

    temp1 = L[low]
    L[low] = minValue
    L[minIndex] = temp1 

    temp2 = L[high]
    L[high] = maxValue

    if low == maxIndex:  
      L[minIndex] = temp2

    else:
      L[maxIndex] = temp2
    
    low += 1
    high -= 1

  return L

#Part 1 Experiment 2 Selection Sort vs Selection Sort 2 Testing
testCasesList = [create_random_list(10,10),create_random_list(50,50),create_random_list(100,100),create_random_list(250,250),
             create_random_list(500,500),create_random_list(750,750),create_random_list(1000,1000),create_random_list(1250,1250),
             create_random_list(1500,1500),create_random_list(2000,2000),create_random_list(2500,2500),create_random_list(3500,3500)
             ]

testTimes = {"selection_sort": [], "selection_sort2": []}
testLengths = []

for testCase in testCasesList:
    selectionSortTime = timeit.timeit(lambda: selection_sort(testCase),number = 10)
    selectionSort2Time = timeit.timeit(lambda: selection_sort2(testCase),number = 10)
    testTimes["selection_sort"].append(selectionSortTime)
    testTimes["selection_sort2"].append(selectionSort2Time)
    testLengths.append(len(testCase))

print("Selection Sort times: {}".format(testTimes["selection_sort"]))
print("Selection Sort 2 times: {}".format(testTimes["selection_sort2"]))

plt.plot(testLengths, testTimes["selection_sort"],label="selection_sort", color = "red")
plt.plot(testLengths,testTimes["selection_sort2"],label="selection_sort2", color = "blue")
plt.xlabel("List Length")
plt.ylabel("Elapsed Time (Seconds)")
plt.title("Selection Sort vs Modified Selection Sort")
plt.legend()

plt.show()

