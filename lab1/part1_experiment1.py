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


# ******************* Bubble sort code *******************

# Traditional Bubble sort
def bubble_sort(L):
    for i in range(len(L)):
        for j in range(len(L) - 1):
            if L[j] > L[j+1]:
                swap(L, j, j+1)


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


# tests
Test_cases = [create_random_list(10, 10), create_random_list(50, 50), create_random_list(100, 100), create_random_list(250, 250), create_random_list(500, 500), create_random_list(
    750, 750), create_random_list(1000, 1000), create_random_list(1250, 1250), create_random_list(1500, 1500), create_random_list(2000, 2000), create_random_list(2500, 2500), create_random_list(3500, 3500)]
times = {"insertion_sort": [], "bubble_sort": [], "selection_sort": []}
list_length = []
for L in Test_cases:
    time3 = timeit.timeit(lambda: insertion_sort(L), number=10)
    time2 = timeit.timeit(lambda: bubble_sort(L), number=10)
    time1 = timeit.timeit(lambda: selection_sort(L), number=10)
    times.get("insertion_sort").append(time1)
    times.get("bubble_sort").append(time2)
    times.get("selection_sort").append(time3)
    list_length.append(len(L))

print(times["bubble_sort"])
# Extract sorting times into a list
sorting_times = [times["insertion_sort"],
                 times["bubble_sort"], times["selection_sort"]]

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

# Show the plot
plt.show()
