import timeit
import matplotlib.pyplot as plt
import random

def swap(L, i, j):
    L[i], L[j] = L[j], L[i]


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


def dual_pivot_quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot1, pivot2 = arr[0], arr[-1]
    left, right, i = 0, len(arr) - 1, 1

    while i <= right:
        if arr[i] < pivot1:
            arr[i], arr[left] = arr[left], arr[i]
            left += 1
        elif arr[i] > pivot2:
            arr[i], arr[right] = arr[right], arr[i]
            right -= 1
            i -= 1  # Recheck the current element after swapping
        i += 1

    # Recursively sort the three segments
    left_segment = dual_pivot_quicksort(arr[:left])
    middle_segment = arr[left:right + 1]
    right_segment = dual_pivot_quicksort(arr[right + 1:])

    return left_segment + middle_segment + right_segment


def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]


# tests
Test_cases = [create_random_list(10, 10), create_random_list(50, 50), create_random_list(100, 100), create_random_list(250, 250), create_random_list(500, 500), create_random_list(
    750, 750)]
times = {"quick_sort": [], "dual_quick_sort": []}
list_length = []
for L in Test_cases:
    time3 = timeit.timeit(lambda: quicksort(L), number=10)
    time2 = timeit.timeit(lambda: dual_pivot_quicksort(L), number=10)
    times.get("quick_sort").append(time3)
    times.get("dual_quick_sort").append(time2)
    list_length.append(len(L))

# Extract sorting times into a list
sorting_times = [times["quick_sort"], times["dual_quick_sort"]]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot list length vs. sorting time for each algorithm
ax.plot(list_length, sorting_times[0], label="quick_sort", color="blue")
ax.plot(list_length, sorting_times[1], label="dual_quick_sort", color="green")

# Set axis labels and a legend
ax.set_xlabel("List Length")
ax.set_ylabel("Time")
ax.set_title("Quick sort variation Time vs. List Length")
ax.legend()

# Show the plot
plt.show()
