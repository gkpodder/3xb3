import min_heap2
import random


class DirectedWeightedGraph:
    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)


def dijkstra(G, source):
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
        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour)
                )
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node

    return dist


# Dijkstra's Approximation Algorithm
def dijkstra_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    Q = min_heap2.MinHeap([])
    nodes = list(G.adj.keys())

    # Relaxation Count dictionary that keeps track of how many times a node has been relaxed
    relaxCount = {}

    # Initialize relaxCount dictionary
    for node in nodes:
        relaxCount[node] = 0

    # Initialize priority queue/heap and distances
    for node in nodes:
        Q.insert(min_heap2.Element(node, float("inf")))
        dist[node] = float("inf")

    Q.decrease_key(source, 0)
    # if k is 0, all distances (except source node) should be infinite
    if k == 0:
        dist[source] = 0
        return dist

    # Meat of the algorithm
    while not Q.is_empty():
        current_element = Q.extract_min()
        current_node = current_element.value
        dist[current_node] = current_element.key

        for neighbour in G.adj[current_node]:
            # if neighbour node has already been relaxed k times, move to next neighbour node
            if relaxCount[neighbour] == k:
                continue

            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour)
                )
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node

                # update neighbour node relaxation count
                relaxCount[neighbour] += 1

    return dist


def bellman_ford(G, source):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    # Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node
    return dist


# bellman ford approximation algorithm
def bellman_ford_approx(G, source, k):
    pred = {}  # Predecessor dictionary. Isn't returned, but here for your understanding
    dist = {}  # Distance dictionary
    nodes = list(G.adj.keys())

    # Relaxation Count Dictionary that keeps track of how many times a node has been relaxed
    relaxCount = {}

    # Initialize Relaxation Count Dictionary
    for node in nodes:
        relaxCount[node] = 0

    # Initialize distances
    for node in nodes:
        dist[node] = float("inf")
    dist[source] = 0

    # if k is 0, all the distances (except for source) should be infinity
    if k == 0:
        return dist

    # Meat of the algorithm
    for _ in range(G.number_of_nodes()):
        for node in nodes:
            for neighbour in G.adj[node]:
                # if neighbour node has already been relaxed k times, skip to next neighbouring node
                if relaxCount[neighbour] == k:
                    continue

                if dist[neighbour] > dist[node] + G.w(node, neighbour):
                    dist[neighbour] = dist[node] + G.w(node, neighbour)
                    pred[neighbour] = node

                    # update neighbour node relaxation count
                    relaxCount[neighbour] += 1
    return dist


def total_dist(dist):
    total = 0
    for key in dist.keys():
        total += dist[key]
    return total


def create_random_complete_graph(n, upper):
    G = DirectedWeightedGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if i != j:
                G.add_edge(i, j, random.randint(1, upper))
    return G


# dijkstra_approx Testing
"""
sampleGraph = create_random_complete_graph(4, 8)
print(dijkstra(sampleGraph, 0))
print("----------------------")
print("Dijkstra Approximation: {}".format(dijkstra_approx(sampleGraph, 0, 3)))
"""

"""
sampleGraph = create_random_complete_graph(4, 8)
print(dijkstra(sampleGraph, 2))
print("----------------------")
print("Approximated dijkstra : {}".format(dijkstra_approx(sampleGraph, 2, 3)))
"""

"""
sampleGraph1 = create_random_complete_graph(7, 8)
print(dijkstra(sampleGraph1, 0))
print("----------------------")
print("Dijkstra Approximation: {}".format(dijkstra_approx(sampleGraph1, 0, 6)))
"""
"""
sampleGraph1 = create_random_complete_graph(7, 8)
print(dijkstra(sampleGraph1, 2))
print("----------------------")
print("Dijkstra Approximation: {}".format(dijkstra_approx(sampleGraph1, 2, 6)))
"""

# bellman_ford_approx Testing
"""
sampleGraph = create_random_complete_graph(5, 10)
print(bellman_ford(sampleGraph, 0))
print("--------------------------")
print("Bellman Ford Approximation: {}".format(bellman_ford_approx(sampleGraph, 0, 4)))
"""
"""
sampleGraph = create_random_complete_graph(5, 10)
print(bellman_ford(sampleGraph, 2))
print("--------------------------")
print("Bellman Ford Approximation: {}".format(bellman_ford_approx(sampleGraph, 2, 7)))
"""
"""
sampleGraph = create_random_complete_graph(8, 10)
print(bellman_ford(sampleGraph, 2))
print("--------------------------")
print("Bellman Ford Approximation: {}".format(bellman_ford_approx(sampleGraph, 2, 0)))
"""
# sampleGraph = create_random_complete_graph(6, 10)
# print(dijkstra(sampleGraph, 2))
# print(bellman_ford(sampleGraph, 2))


# Assumes G represents its nodes as integers 0,1,...,(n-1)
def mystery(G):
    n = G.number_of_nodes()
    d = init_d(G)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if d[i][j] > d[i][k] + d[k][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d
