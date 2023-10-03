import random
import timeit
import matplotlib.pyplot as plt

# Create a random list length "length" containing whole numbers between 0 and max_value inclusive
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

# I have created this function to make the sorting algorithm code read easier
def swap(L, i, j):
    L[i], L[j] = L[j], L[i]

# Creates a near sorted list by creating a random list, sorting it, then doing a random number of swaps
def create_near_sorted_list(length, max_value, swaps):
    L = create_random_list(length, max_value)
    L.sort()
    for _ in range(swaps):
        r1 = random.randint(0, length - 1)
        r2 = random.randint(0, length - 1)
        swap(L, r1, r2)
    return L
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
length=500
max_val=800

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


#**********EXPERIMENT 3****************


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
