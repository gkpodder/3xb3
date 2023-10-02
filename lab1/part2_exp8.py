import random
import timeit
import matplotlib.pyplot as plt

#Note: This experiment contains some functions and sorting algorithms from the bad_sorts.py and good_sorts.py file

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

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


# ************ Quick Sort ************
def quicksort(L):
    copy = quicksort_copy(L)
    for i in range(len(L)):
        L[i] = copy[i]


def quicksort_copy(L):
    if len(L) < 2:
        return L
    pivot = L[0]
    left, right = [], []
    for num in L[1:]:
        if num < pivot:
            left.append(num)
        else:
            right.append(num)
    return quicksort_copy(left) + [pivot] + quicksort_copy(right)

# *************************************

# ************ Merge Sort *************

def mergesort(L):
    if len(L) <= 1:
        return
    mid = len(L) // 2
    left, right = L[:mid], L[mid:]

    mergesort(left)
    mergesort(right)
    temp = merge(left, right)

    for i in range(len(temp)):
        L[i] = temp[i]


def merge(left, right):
    L = []
    i = j = 0

    while i < len(left) or j < len(right):
        if i >= len(left):
            L.append(right[j])
            j += 1
        elif j >= len(right):
            L.append(left[i])
            i += 1
        else:
            if left[i] <= right[j]:
                L.append(left[i])
                i += 1
            else:
                L.append(right[j])
                j += 1
    return L

# *************************************

#Part 2 Experiment 8 Testing Insertion Sort vs Merge Sort vs Quick Sort
'''
testCasesList = [create_random_list(10,10),create_random_list(25,25),create_random_list(50,50),create_random_list(75,75),
                create_random_list(100,100),create_random_list(125,125),create_random_list(150,150),
                create_random_list(175,175),create_random_list(200,200)]
'''


testCasesList = [create_random_list(10,10),create_random_list(25,25),create_random_list(50,50),create_random_list(75,75),
                create_random_list(100,100),create_random_list(125,125),create_random_list(150,150),
                create_random_list(175,175),create_random_list(200,200),create_random_list(225,225),
                create_random_list(250,250),create_random_list(275,275),create_random_list(300,300),
                create_random_list(325,325),create_random_list(350,350),create_random_list(375,375),
                create_random_list(400,400),create_random_list(425,425),create_random_list(450,450),
                create_random_list(475,475),create_random_list(500,500),create_random_list(525,525),
                create_random_list(550,550),create_random_list(575,575),create_random_list(600,600),
                create_random_list(625,625),create_random_list(650,650),create_random_list(675,675),
                create_random_list(700,700),create_random_list(725,725),create_random_list(750,750)]

testTimes = {"insertion_sort": [], "quicksort": [], "mergesort": []}
testLengths = []

for testCase in testCasesList:
    insertionSortTime = timeit.timeit(lambda:insertion_sort(testCase),number = 10)
    quickSortTime = timeit.timeit(lambda:quicksort(testCase),number = 10)
    mergeSortTime = timeit.timeit(lambda:mergesort(testCase),number = 10)
    testTimes["insertion_sort"].append(insertionSortTime)
    testTimes["quicksort"].append(quickSortTime)
    testTimes["mergesort"].append(mergeSortTime)
    testLengths.append(len(testCase))

print("Insertion Sort times: {}".format(testTimes["insertion_sort"]))
print("Quick Sort times: {}".format(testTimes["quicksort"]))
print("Merge Sort times: {}".format(testTimes["mergesort"]))

plt.plot(testLengths,testTimes["insertion_sort"],label="insertion_sort", color = "red")
plt.plot(testLengths,testTimes["quicksort"],label="quicksort", color = "blue")
plt.plot(testLengths,testTimes["mergesort"],label="mergesort", color = "green")
plt.xlabel("List Length")
plt.ylabel("Elapsed Time (Seconds)")
plt.title("Insertion Sort vs Merge Sort vs Quick Sort")
plt.legend()

plt.show() 
