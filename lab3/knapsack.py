import random
import timeit
import matplotlib.pyplot as plt
import numpy as np


def generate_random_item_set(num_items=20, weight_range=(10, 100), value_range=(1000, 2000)):
    items = []
    for _ in range(num_items):
        weight = random.randint(weight_range[0], weight_range[1])
        value = random.randint(value_range[0], value_range[1])
        items.append((weight, value))
    return items


def ks_brute_force(items: list[tuple[int]], capacity: int) -> int:
    def generate_combinations(index, current_combination):
        nonlocal max_value

        if index == len(items):
            # Calculate total weight and value for the current combination
            total_weight = sum(
                current_combination[i] * items[i][0] for i in range(len(items)))
            total_value = sum(
                current_combination[i] * items[i][1] for i in range(len(items)))

            # Check if the combination is feasible (weight does not exceed capacity)
            if total_weight <= capacity and total_value > max_value:
                max_value = total_value
            return

        # Explore the two possibilities for each item: include or exclude
        # Exclude item at index
        generate_combinations(index + 1, current_combination + [0])
        # Include item at index
        generate_combinations(index + 1, current_combination + [1])

    max_value = 0
    generate_combinations(0, [])
    return max_value


def ks_rec(items: list[tuple[int]], capacity: int) -> int:
    # Base case: If either the items list is empty or the capacity is 0, the value is 0
    if not items or capacity == 0:
        return 0

    # Get the last item in the list
    last_item = items[-1]

    # Check if the weight of the last item exceeds the current capacity
    if last_item[0] > capacity:
        # Exclude the last item and recursively solve for the remaining items
        return ks_rec(items[:-1], capacity)
    else:
        # Choose the maximum value between including and excluding the last item
        include_last = ks_rec(
            items[:-1], capacity - last_item[0]) + last_item[1]
        exclude_last = ks_rec(items[:-1], capacity)
        return max(include_last, exclude_last)

# Brute force vs Recursive experiment code:
# def time_experiment():
#     item_set_sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17]
#     brute_force_times = []
#     recursive_times = []

#     for size in item_set_sizes:
#         items = generate_random_item_set(num_items=size)

#         # Measure time for brute-force solution
#         print("hi")
#         brute_force_time = timeit.timeit(
#             lambda: ks_brute_force(items, 1000), number=10)
#         brute_force_times.append(brute_force_time)
#         print("bye")
#         # Measure time for recursive solution
#         recursive_time = timeit.timeit(
#             lambda: ks_rec(items, 1000), number=10)
#         recursive_times.append(recursive_time)

#     return brute_force_times, recursive_times


# def plot_results(item_set_sizes, brute_force_times, recursive_times):
#     print(item_set_sizes, brute_force_times, recursive_times)
#     plt.plot(item_set_sizes, brute_force_times, label='Brute Force')
#     plt.plot(item_set_sizes, recursive_times, label='Recursive')
#     plt.xlabel('Item Set Size')
#     plt.ylabel('Execution Time (seconds)')
#     plt.title('Knapsack Problem: Brute Force vs. Recursive')
#     plt.legend()
#     plt.show()


# brute_force_times, recursive_times = time_experiment()
# item_set_sizes = [1, 3, 5, 7, 9, 11, 13, 15, 17]
# plot_results(item_set_sizes, brute_force_times, recursive_times)

def ks_bottom_up(items: list[tuple[int]], capacity: int) -> int:
    n = len(items)
    # table to store subproblem results
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            weight, value = items[i - 1]
            if weight <= w:
                # Choose the maximum value between including and excluding the current item
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)
            else:
                # If current item's weight exceeds the remaining capacity, exclude it
                dp[i][w] = dp[i - 1][w]

    return dp[n][capacity]


def ks_top_down(items: list[tuple[int]], capacity: int) -> int:
    n = len(items)
    memo = dict()

    def ks_recursive(i, remaining_capacity):
        # Base case
        if i == 0 or remaining_capacity == 0:
            return 0

        # Check if the result for this subproblem is already memoized
        if (i, remaining_capacity) in memo:
            return memo[(i, remaining_capacity)]

        # current item
        weight, value = items[i - 1]

        # If the current item's weight exceeds the remaining capacity, skip it
        if weight > remaining_capacity:
            result = ks_recursive(i - 1, remaining_capacity)
        else:
            # Choose the maximum value between including and excluding the current item
            include_current = value + \
                ks_recursive(i - 1, remaining_capacity - weight)
            exclude_current = ks_recursive(i - 1, remaining_capacity)
            result = max(include_current, exclude_current)

        # Memoize the result for this subproblem
        memo[(i, remaining_capacity)] = result
        return result

    return ks_recursive(n, capacity)


# BU vs TD experiment:
# def time_experiment():
#     item_set_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
#     brute_force_times = []
#     recursive_times = []

#     for size in item_set_sizes:
#         items = generate_random_item_set(num_items=size)

#         # Measure time for brute-force solution
#         print("hi")
#         brute_force_time = timeit.timeit(
#             lambda: ks_top_down(items, int((100+10)/2)), number=10)
#         brute_force_times.append(brute_force_time)
#         print("bye")
#         # Measure time for recursive solution
#         recursive_time = timeit.timeit(
#             lambda: ks_bottom_up(items, int((100+10)/2)), number=10)
#         recursive_times.append(recursive_time)

#     return brute_force_times, recursive_times


# def plot_results(item_set_sizes, brute_force_times, recursive_times):
#     print(item_set_sizes, brute_force_times, recursive_times)
#     plt.plot(item_set_sizes, brute_force_times, label='top down')
#     plt.plot(item_set_sizes, recursive_times, label='bottom up')
#     plt.xlabel('Item Set Size')
#     plt.ylabel('Execution Time (seconds)')
#     plt.title('Knapsack Problem: BU vs. TD')
#     plt.legend()
#     plt.show()


# brute_force_times, recursive_times = time_experiment()
# item_set_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# plot_results(item_set_sizes, brute_force_times, recursive_times)
