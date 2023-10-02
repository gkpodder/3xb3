import random
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

def bottum_up_mergesort(L):
    window_size=1
    while(len(L)>window_size):
        for i in range(0,len(L),window_size*2):
            leftStart=i
            leftEnd=min(i+window_size,len(L))
            rightEnd=min(leftEnd+window_size,len(L))
            L[leftStart:rightEnd]=merge(L[leftStart:leftEnd],L[leftEnd:rightEnd])
        window_size=window_size*2
    return L
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

#Part 2 Experiment 7 Merge Sort v.s. Bottum Up Merge Sort Testing
testCasesList = [create_random_list(10,10),create_random_list(50,50),create_random_list(100,100),create_random_list(250,250),
             create_random_list(500,500),create_random_list(750,750),create_random_list(1000,1000),create_random_list(1250,1250),
             create_random_list(1500,1500),create_random_list(2000,2000),create_random_list(2500,2500),create_random_list(3500,3500)
             ]
testTimes = {"mergesort": [], "bottum_up_mergesort": []}
testLengths = []

for testCase in testCasesList:
    mergeSortTime = timeit.timeit(lambda: mergesort(testCase),number = 10)
    mergeSort2Time = timeit.timeit(lambda: bottum_up_mergesort(testCase),number = 10)
    testTimes["mergesort"].append(mergeSortTime)
    testTimes["bottum_up_mergesort"].append(mergeSort2Time)
    testLengths.append(len(testCase))

print("Merge Sort times: {}".format(testTimes["mergesort"]))
print("Bottum Up Merge Sort times: {}".format(testTimes["bottum_up_mergesort"]))

plt.plot(testLengths, testTimes["mergesort"],label="mergesort", color = "red")
plt.plot(testLengths,testTimes["bottum_up_mergesort"],label="bottum_up_mergesort", color = "blue")
plt.xlabel("List Length")
plt.ylabel("Elapsed Time")
plt.title("Merge Sort vs Bottom Up Bubble Sort")
plt.legend()

plt.show()
