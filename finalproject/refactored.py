import min_heap2
from typing import Optional, Tuple, List, Dict


class Graph:

    def __init__(self) -> None:
        self.adj = {}

    def get_adj_nodes(self, node: int) -> list:
        return self.adj[node]

    def add_node(self, node: int):
        self.adj[node] = []

    def add_edge(self, start: int, end: int):
        pass

    def get_num_of_nodes(self) -> int:
        return len(self.adj)


class WeightedGraph(Graph):
    def __init__(self):
        super().__init__()
        self.weights = {}

    def add_edge(self, start: int, end: int, weight: float):
        super().add_edge(start, end)
        if end not in self.adj[start]:
            self.adj[start].append(end)
        self.weights[(start, end)] = weight

    def w(self, node1: int, node2: int) -> float:
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return self.weights[(node1, node2)]


class HeuristicGraph(WeightedGraph):
    def __init__(self):
        super().__init__()
        self.heuristic = {}

    # def set_heuristic(self, heuristic: dict):
    #     self.heuristic = heuristic

    def get_heuristic(self) -> Dict[int, float]:
        return self.heuristic


class SPAlgorithm:
    def calc_sp(self, graph: Graph, source: int, dest: int) -> float:
        pass


class Dijkstra(SPAlgorithm):
    def calc_sp(self, G: Graph, source: int, dest: Optional[int] = None) -> float:
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
                        neighbour, dist[current_node] +
                        G.w(current_node, neighbour)
                    )
                    dist[neighbour] = dist[current_node] + \
                        G.w(current_node, neighbour)
                    pred[neighbour] = current_node

        if dest is not None:
            return dist[dest]
        else:
            return dist


class Bellman_Ford(SPAlgorithm):
    def calc_sp(self, G: Graph, source: int, dest: Optional[int] = None) -> float:
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
        return dist[dest]


class A_Star(SPAlgorithm):
    def calc_sp(self, G: Graph, source: int, dest: Optional[int] = None) -> float:
        try:
            heuristic = G.get_heuristic()
        except AttributeError:
            raise ValueError("Heuristic not set")
        return a_star(G, source, dest, heuristic)[0][dest]


def a_star(G, source, goal, h):
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
                h[(current_node, neighbour)]  # A* specific part

            if tentative_f < dist[neighbour]:
                Q.decrease_key(neighbour, tentative_f)
                dist[neighbour] = tentative_f
                pred[neighbour] = current_node

    return dist, pred


class ShortPathFinder:
    def __init__(self):
        self.graph = None
        self.algorithm = None

    def calc_short_path(self, source: int, dest: int) -> float:
        if not self.algorithm:
            raise ValueError("Algorithm not set")
        return self.algorithm.calc_sp(self.graph, source, dest)

    def set_graph(self, graph: Graph):
        self.graph = graph

    def set_algorithm(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm


# test case
g = WeightedGraph()
g.add_node(1)
g.add_node(2)
g.add_edge(1, 2, 3.5)
print(g.weights)
