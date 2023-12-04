from final_project_part1 import DirectedWeightedGraph
import min_heap2
import math
import pandas as pd

stationslist=pd.read_csv("london_stations.csv")
connectionslist=pd.read_csv("london_connections.csv")


graph_of_London=DirectedWeightedGraph()

for i, r in stationslist.iterrows():
    graph_of_London.add_node(r['id'])

for i, r in connectionslist.iterrows():
    graph_of_London.add_edge(r['station1'],r['station2'],r['time'])

station_coordinates = {r['id']:(r['latitude'],r['longitude']) for i, r in stationslist.iterrows()}

def heuristic(node, goal):
    # It should estimate the cost from the current node to the goal node
    # The heuristic function should be admissible (never overestimates the true cost)
    # use Euclidean distance as a heuristic function
    #distance=sqrt((x2-x1)^2+(y2-y1)^2)
    node_coords = station_coordinates[node]
    goal_coords = station_coordinates[goal]
    return math.sqrt((goal_coords[0] - node_coords[0])**2 + (goal_coords[1] - node_coords[1])**2)
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


start_id = 13  # Replace with actual source station ID
goal_id = 279   # Replace with actual goal station ID
generated_path = a_star(graph_of_London, start_id, goal_id)

print(generated_path)

