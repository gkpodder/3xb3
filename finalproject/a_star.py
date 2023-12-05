from final_project_part1 import DirectedWeightedGraph
import min_heap2
import math
import pandas as pd
from final_project_part1 import dijkstra as dj
import timeit
import matplotlib.pyplot as plt

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

def dijkstra(G, source, goal):
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

        # Check if the goal node is reached
        if current_node == goal:
            # Reconstruct the shortest path from goal to source
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = pred.get(current_node, None)
            return dist[goal], path  # Return distance and path to goal

        for neighbour in G.adj[current_node]:
            if dist[current_node] + G.w(current_node, neighbour) < dist[neighbour]:
                Q.decrease_key(
                    neighbour, dist[current_node] + G.w(current_node, neighbour)
                )
                dist[neighbour] = dist[current_node] + G.w(current_node, neighbour)
                pred[neighbour] = current_node

    return dist  # Return the distance dictionary for all nodes if the goal isn't reached


# start_id = 73  # Replace with actual source station ID
# goal_id = 265  # Replace with actual goal station ID
# generated_path1 = a_star(graph_of_London, start_id, goal_id)
# #print(graph_of_London.adj)
# generated_path2= dijkstra(graph_of_London,start_id,goal_id)
# print("A*:\n")
# print(generated_path1)
# print("Djikstra:\n")
# print(generated_path2)


stations = [r['id'] for i, r in stationslist.iterrows()]

def experiment_every_pair():
    #i=0
    dijkstra_times = []
    a_star_times = []

    for source in stations:
        dijkstra_time = 0
        a_star_time = 0
        #i+=1
        #print(i)
        #if(i==20): 
         #   break

        for goal in stations:
            if source == goal:
                continue
            start = timeit.default_timer()
            a_star(graph_of_London, source, goal)
            a_star_time += timeit.default_timer() - start
            start = timeit.default_timer()
            dijkstra(graph_of_London, source, goal)
            dijkstra_time += timeit.default_timer() - start
        dijkstra_times.append(dijkstra_time)
        a_star_times.append(a_star_time)
    plt.plot(dijkstra_times, label="Dijkstra")
    plt.plot(a_star_times, label="A*")
    plt.legend()
    plt.xlabel("Station Number")
    plt.ylabel("Time (seconds)")
    plt.title("Time to Calculate Shortest Path Pairs From Every Station")
    plt.show()
    
experiment_every_pair()

def find_lines():
    lines_dictionary = {}
    connections = pd.read_csv("london_connections.csv")
    for line, setting in connections.groupby('line'):
        lines_dictionary[line] = set(setting['station1']).union(set(setting['station2']))
    return lines_dictionary

def experiment_same_line():
    dijkstra_times = []
    astar_times = []
    
    lines = find_lines() 
    m_station_line = max(lines, key=lambda line_id: len(lines[line_id]))
    stations_sorted = sorted(lines[m_station_line])
    
    for station_start in stations_sorted:
        time_dijkstra = 0
        time_a_star = 0
        
        for station_end in lines[m_station_line]:
            if station_start == station_end:
                continue 

            start_time = timeit.default_timer()
            a_star(graph_of_London, station_start, station_end)
            time_a_star += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            dijkstra(graph_of_London, station_start, station_end)
            time_dijkstra += timeit.default_timer() - start_time

        astar_times.append(time_a_star)
        dijkstra_times.append(time_dijkstra)

    plt.figure(figsize=(10, 5))
    plt.plot(astar_times, label="A* Algorithm")
    plt.plot(dijkstra_times, label="Dijkstra's Algorithm")
    plt.xlabel("Iteration for Station Pairs")
    plt.ylabel("Total Time (seconds)")
    plt.title(f"Shortest Path Time On Line {m_station_line}")
    plt.legend()
    plt.show()

experiment_same_line()



def diff_lines():
    dijkstra_times = []
    a_star_times = []
    londonGraph = graph_of_London
    lines = find_lines()
    line1_stations = list(lines[1])
    line4_stations = list(lines[4])

    for s1 in line1_stations:
        dijkstra_time = 0
        a_star_time = 0

        for s2 in line4_stations:
            if s1 == s2:
                continue
            start = timeit.default_timer()
            a_star(londonGraph, s1, s2)
            a_star_time += timeit.default_timer() - start
            start = timeit.default_timer()
            dijkstra(londonGraph, s1, s2)
            dijkstra_time += timeit.default_timer() - start
        a_star_times.append(a_star_time)
        dijkstra_times.append(dijkstra_time)
    xticks = range(len(line1_stations))
    plt.plot(xticks, dijkstra_times, label="Dijkstra")
    plt.plot(xticks, a_star_times, label="A*")
    plt.xlabel("Station Index in Line 1")
    plt.ylabel("Total Time(seconds)")
    plt.title("Time to Calculate Shortest Path Pairs Between line 1 and 4")
    plt.legend()
    plt.tight_layout()
    plt.show()
diff_lines()
