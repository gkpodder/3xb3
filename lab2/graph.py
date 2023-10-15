from collections import deque


# Undirected graph using an adjacency list


class Graph:

    def __init__(self, n):
        self.adj = {}
        for i in range(n):
            self.adj[i] = []

    def are_connected(self, node1, node2):
        return node2 in self.adj[node1]

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self):
        self.adj[len(self.adj)] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adj[node2]:
            self.adj[node1].append(node2)
            self.adj[node2].append(node1)

    def number_of_nodes():
        return len()


# Breadth First Search
def BFS(G, node1, node2):
    Q = deque([node1])
    marked = {node1: True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
    while len(Q) != 0:
        current_node = Q.popleft()
        for node in G.adj[current_node]:
            if node == node2:
                return True
            if not marked[node]:
                Q.append(node)
                marked[node] = True
    return False


# Depth First Search
def DFS(G, node1, node2):
    S = [node1]
    marked = {}
    for node in G.adj:
        marked[node] = False
    while len(S) != 0:
        current_node = S.pop()
        if not marked[current_node]:
            marked[current_node] = True
            for node in G.adj[current_node]:
                if node == node2:
                    return True
                S.append(node)
    return False


# Use the methods below to determine minimum Vertex Covers
def add_to_each(sets, element):
    copy = sets.copy()
    for set in copy:
        set.append(element)
    return copy


def power_set(set):
    if set == []:
        return [[]]
    return power_set(set[1:]) + add_to_each(power_set(set[1:]), set[0])


def is_vertex_cover(G, C):
    for start in G.adj:
        for end in G.adj[start]:
            if not (start in C or end in C):
                return False
    return True


def MVC(G):
    nodes = [i for i in range(G.get_size())]
    subsets = power_set(nodes)
    min_cover = nodes
    for subset in subsets:
        if is_vertex_cover(G, subset):
            if len(subset) < len(min_cover):
                min_cover = subset
    return min_cover


def BFS2(G, node1, node2):
    visited = set()
    queue = deque([(node1, [node1])])

    while queue:
        current_node, path = queue.popleft()
        visited.add(current_node)

        if current_node == node2:
            return path

        for neighbor in G.adjacent_nodes(current_node):
            if neighbor not in visited:
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

    return []


def DFS2(G, node1, node2, visited=None):
    if visited is None:
        visited = set()

    visited.add(node1)

    if node1 == node2:
        return [node1]

    for neighbor in G.adjacent_nodes(node1):
        if neighbor not in visited:
            path = DFS2(G, neighbor, node2, visited)
            if path:
                return [node1] + path

    return []

#BFS3 Predecessor Dictionary
def BFS3(G,node1):
    queue = deque([node1])
    predDictionary = {}
    marked = {node1:True}
    for node in G.adj:
        if node != node1:
            marked[node] = False
            
    while len(queue) != 0:
        currentNode = queue.popleft()

        for adjacentNode in G.adj[currentNode]:
            if marked[adjacentNode] == False:
                queue.append(adjacentNode)
                marked[adjacentNode] = True
                predDictionary[adjacentNode] = currentNode 
    
    return predDictionary

'''
# test case
# Create a adj_list with 6 nodes and 7 edges
g = Graph(11)
g.add_edge(0, 1)
g.add_edge(1, 4)
g.add_edge(1, 10)
g.add_edge(4, 10)
g.add_edge(4, 3)
g.add_edge(3, 5)

# Test BFS2 with node1 = 0 and node2 = 5
path1 = BFS2(g, 0, 5)
path2 = DFS2(g, 0, 5)

print(path1, " ", path2)
'''

#BFS3Testing
#Graph
testGraph = Graph(6)
testGraph.add_edge(0,1)
testGraph.add_edge(0,2)
testGraph.add_edge(1,3)
testGraph.add_edge(2,3)
testGraph.add_edge(2,4)
testGraph.add_edge(4,3)
testGraph.add_edge(3,5)
pred1 = BFS3(testGraph,0)
pred2 = BFS3(testGraph,1)
#print(pred1)
#print(pred2)

test2 = Graph(7)
test2.add_edge(0,1)
test2.add_edge(0,2)
test2.add_edge(0,3)
test2.add_edge(1,2)
test2.add_edge(2,3)
test2.add_edge(1,5)
test2.add_edge(2,4)
test2.add_edge(3,6)
test2.add_edge(5,4)
test2.add_edge(4,6)
pred3 = BFS3(test2,3)
pred4 = BFS3(test2,4)
#print(pred3)
#print(pred4)

test3 = Graph(9)
test3.add_edge(0,3)
test3.add_edge(1,2)
test3.add_edge(2,3)
test3.add_edge(3,6)
test3.add_edge(3,4)
#print(test3.adj)
pred5 = BFS3(test3,3)
pred6 = BFS3(test3,0)
pred7 = BFS3(test3,5)
print(pred5)
print(pred6)
print(pred7)