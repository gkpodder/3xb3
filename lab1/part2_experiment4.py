import timeit
import math
import random
import matplotlib.pyplot as plt

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

# ************* Heap Sort *************


def heapsort(L):
    heap = Heap(L)
    for _ in range(len(L)):
        heap.extract_max()


class Heap:
    length = 0
    data = []

    def __init__(self, L):
        self.data = L
        self.length = len(L)
        self.build_heap()

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i)

    def heapify(self, i):
        largest_known = i
        if self.left(i) < self.length and self.data[self.left(i)] > self.data[i]:
            largest_known = self.left(i)
        if self.right(i) < self.length and self.data[self.right(i)] > self.data[largest_known]:
            largest_known = self.right(i)
        if largest_known != i:
            self.data[i], self.data[largest_known] = self.data[largest_known], self.data[i]
            self.heapify(largest_known)

    def insert(self, value):
        if len(self.data) == self.length:
            self.data.append(value)
        else:
            self.data[self.length] = value
        self.length += 1
        self.bubble_up(self.length - 1)

    def insert_values(self, L):
        for num in L:
            self.insert(num)

    def bubble_up(self, i):
        while i > 0 and self.data[i] > self.data[self.parent(i)]:
            self.data[i], self.data[self.parent(
                i)] = self.data[self.parent(i)], self.data[i]
            i = self.parent(i)

    def extract_max(self):
        self.data[0], self.data[self.length -
                                1] = self.data[self.length - 1], self.data[0]
        max_value = self.data[self.length - 1]
        self.length -= 1
        self.heapify(0)
        return max_value

    def left(self, i):
        return 2 * (i + 1) - 1

    def right(self, i):
        return 2 * (i + 1)

    def parent(self, i):
        return (i + 1) // 2 - 1

    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.data[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s

# *************************************


def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]


# tests
Test_cases = [create_random_list(n, n) for n in range(10, 1000, 10)]
times = {"quick_sort": [], "merge_sort": [], "heap_sort": []}
list_length = []
for L in Test_cases:
    time1 = timeit.timeit(lambda: quicksort(L), number=20)
    time2 = timeit.timeit(lambda: mergesort(L), number=20)
    time3 = timeit.timeit(lambda: heapsort(L), number=20)
    times.get("quick_sort").append(time1)
    times.get("merge_sort").append(time2)
    times.get("heap_sort").append(time3)
    list_length.append(len(L))

# Extract sorting times into a list
sorting_times = [times["quick_sort"], times["merge_sort"], times["heap_sort"]]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot list length vs. sorting time for each algorithm
ax.plot(list_length, sorting_times[0], label="quick", color="blue")
ax.plot(list_length, sorting_times[1], label="merge", color="green")
ax.plot(list_length, sorting_times[2], label="heap", color="red")

# Set axis labels and a legend
ax.set_xlabel("List Length")
ax.set_ylabel("Time")
ax.set_title("Sorting Algo Time vs. List Length")
ax.legend()

# Show the plot
plt.show()
