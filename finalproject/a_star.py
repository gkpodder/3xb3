from final_project_part1 import DirectedWeightedGraph
import min_heap2
import math
import pandas as pd
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

def heuristic(station1, station2, coordinates):
    coord1 = coordinates[station1]
    coord2 = coordinates[station2]
    return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

def make_h(stations,station_coordinates):
    precalculated_heuristics = {}
    for source in stations:
        for goal in stations:
            if source != goal:
                precalculated_heuristics[(source, goal)] = heuristic(source, goal, station_coordinates)
    return precalculated_heuristics


def a_star(G, source, goal,h):
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
            tentative_f = tentative_g + h[(current_node, neighbour)]  # A* specific part

            if tentative_f < dist[neighbour]:
                Q.decrease_key(neighbour, tentative_f)
                dist[neighbour] = tentative_f
                pred[neighbour] = current_node

    return dist,pred

def dj_2(G, source,goal):
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

    return dist[goal]


# start_id = 73  # Replace with actual source station ID
# goal_id = 265  # Replace with actual goal station ID
# generated_path1 = a_star(graph_of_London, start_id, goal_id)
# #print(graph_of_London.adj)
# generated_path2= dijkstra(graph_of_London,start_id,goal_id)
# print("A*:\n")
# print(generated_path1)
# print("Djikstra:\n")
# print(generated_path2)

def dj(G, source):
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

stations = [r['id'] for i, r in stationslist.iterrows()]
def timing_exp(stations,graph_of_London,h):
    for source in stations:
        for goal in stations:
                if source == goal:
                    continue
                a_star(graph_of_London, source, goal,h)
            
def dj_timing(graph,stations):
    
    for source in stations:
        dj(graph, source)

def experiment_every_pair():
    h = make_h(stations,station_coordinates)
    result_a_star=timeit.timeit(lambda: timing_exp(stations,graph_of_London,h),number=1)
    result_dj = timeit.timeit(lambda: dj_timing(graph_of_London,stations),number=1)
    print(result_a_star)
    print(result_dj)
    
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
    h = make_h(stations,station_coordinates)
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
            a_star(graph_of_London, station_start, station_end,h)

            time_a_star += timeit.default_timer() - start_time

            start_time = timeit.default_timer()
            dj_2(graph_of_London, station_start, station_end)
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
    h = make_h(stations,station_coordinates)

    for s1 in line1_stations:
        dijkstra_time = 0
        a_star_time = 0

        for s2 in line4_stations:
            if s1 == s2:
                continue
            start = timeit.default_timer()
            a_star(londonGraph, s1, s2,h)
            a_star_time += timeit.default_timer() - start
            start = timeit.default_timer()
            dj_2(londonGraph, s1, s2)
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

def experiment_multiple_transfers():
    dijkstra_times = []
    a_star_times = []
    h = make_h(stations, station_coordinates)
    lines = find_lines()

    transfer_stations_pairs = [(132, 268), (165, 216), (206, 217), (266, 268), (68, 283), (16, 93), (53, 160), (144, 278), (185, 279), (218, 227)] 
    for (source, destination) in transfer_stations_pairs:
        time_dijkstra = 0
        time_a_star = 0
        start_time = timeit.default_timer()
        a_star(graph_of_London, source, destination, h)
        time_a_star += timeit.default_timer() - start_time
        start_time = timeit.default_timer()
        dj_2(graph_of_London, source, destination)
        time_dijkstra += timeit.default_timer() - start_time
        a_star_times.append(time_a_star)
        dijkstra_times.append(time_dijkstra)
    plt.figure(figsize=(10, 5))
    plt.plot(a_star_times, label="A* Algorithm")
    plt.plot(dijkstra_times, label="Dijkstra's Algorithm")
    plt.xlabel("Pair Index")
    plt.ylabel("Total Time (seconds)")
    plt.title("Performance Comparison of A* and Dijkstra for Stations Requiring Transfers")
    plt.legend()
    plt.show()

experiment_multiple_transfers()

