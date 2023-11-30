from final_project_part1 import DirectedWeightedGraph
import min_heap2


def heuristic(node, goal):
    # It should estimate the cost from the current node to the goal node
    # The heuristic function should be admissible (never overestimates the true cost)
    # use Euclidean distance as a heuristic function
    pass


def a_star(G, source, goal):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    Q = min_heap2.MinHeap([])
    nodes = list(G.adj.keys())

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap2.Element(node, float("inf")))
        dist[node] = float("inf")
    Q.decrease_key(source, 0)

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key

        if current_node == goal:
            break  # Exit early if the goal is reached

        for neighbour in G.adj[current_node]:
            tentative_g = dist[current_node] + G.w(current_node, neighbour)
            tentative_f = tentative_g + \
                heuristic(neighbour, goal)  # A* specific part

            if tentative_f < dist[neighbour]:
                Q.decrease_key(neighbour, tentative_f)
                dist[neighbour] = tentative_f
                pred[neighbour] = current_node

    return dist
