import random
import timeit
import matplotlib.pyplot as plt

#Note: This experiment contains some functions and the selection sort 
# algorithm from the bad_sorts.py file and the selection_sort2 function from
# selection_sort2.py file

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

# I have created this function to make the sorting algorithm code read easier
def swap(L, i, j):
    L[i], L[j] = L[j], L[i]

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

