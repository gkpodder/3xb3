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


# Example usage:
items = [(2, 10), (3, 15), (5, 25)]
capacity = 8
result = ks_rec(items, capacity)
print(result)
