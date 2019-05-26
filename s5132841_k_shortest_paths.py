"""
K-SHORTEST-PATH PROBLEM v1.0
Author: Lucas Geurtjens (s5132841)
Date: 26/05/2019

REFERENCES/CREDITS:
[1]
Dijkstra's SP implementation and graph structure was heavily influenced by 'Ben Alex Keen' (2017)
URL : http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/

[2]
The alternate k-shortest path implementation was loosely inspired by the Yen's algorithm created by 'Jin Y. Yen' (1971)
URL: https://en.wikipedia.org/wiki/Yen%27s_algorithm
"""

from collections import defaultdict
import copy
import math
import os
import sys
import time


class Graph:
    def __init__(self):
        self.edges = defaultdict(list)  # Dictionary of all edges
        self.weights = {}  # Dictionary of all weights

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)  # Set edge of node
        self.weights[(from_node, to_node)] = weight  # Set weight of node


def dijkstra(graph, start, goal):
    """
    Performs a Dijkstra's shortest path search
    :param graph: A graph of edges and weights
    :param start: Starting node
    :param goal: Goal node to reach
    :return: Path list of the optimal shortest path and the distance cost to reach the goal node
    """
    dist_costs = {}  # Costs to travel to each node. (Initialised with starting node)
    dist_costs.update({start: (None, 0)})  # Update distance costs dict with starting node
    current_node = start  # Initialise starting node to the current node
    explored = set()  # Set for storing already explored nodes

    while current_node != goal:  # While we haven't reached the goal node

        explored.add(current_node)
        child_paths = graph.edges[current_node]  # Edges connected to current node we can traverse
        current_weight = dist_costs[current_node][1]  # Grab the weight of current node

        for child_node in child_paths:  # For all possible child nodes of the current node...

            # Create a new weight by adding its cost and the parent's current distance cost
            new_weight = graph.weights[(current_node, child_node)] + current_weight

            # If this is our first time visiting the node, add it to the dist costs dictionary
            if child_node not in dist_costs:
                dist_costs.update({child_node: (current_node, new_weight)})

            # Otherwise if we have, see if we can relax the weight of the node.
            else:
                min_weight = dist_costs[child_node][1]
                if new_weight < min_weight:  # If the new weight is smaller, we can relax the cost
                    dist_costs.update({child_node: (current_node, new_weight)})

        # Get a dictionary of all possible node paths we can take that haven't already been visited
        nodes_to_check = {}
        for node in dist_costs:
            if node not in explored:
                nodes_to_check.update({node: dist_costs[node]})

        # If there are no possible node paths and we aren't yet at the goal node -> then the route isn't possible.
        if not nodes_to_check:
            return "Sorry, the input route was not possible."

        # Otherwise, find the lowest cost next node to check (like a PQ dequeue)
        else:
            # The next current node is one with a minimum cost in the nodes to check dictionary
            current_node = min(nodes_to_check, key=lambda x: nodes_to_check[x][1])

    # Now that the goal node is reached, we want to trace back the optimal path and distance cost we found
    optimal_path = []
    optimal_cost = dist_costs[goal][1]  # Optimal cost will always be the distance of the goal node.

    # While we haven't reached the starting node...
    while current_node is not None:
        optimal_path.append(current_node)  # Append the current node
        child_node = dist_costs[current_node][0]  # Grab child of current node
        current_node = child_node  # Set current node to be the child

    optimal_path = optimal_path[::-1]  # Reverse the path (since we went from end -> start)

    return optimal_path, optimal_cost


def alt_shortest_paths(shortest_path, known_graph, k):
    """
    Finds the alternate (k - 1) shortest path approximations.
    :param shortest_path: Current known shortest route from the Dijkstra algorithm
    :param known_graph: Graph of the edges and weights
    :param k: Current k-value
    :return: A list of k - 1 approximate shortest route costs
    """
    k_shortest = []  # Will store the alternate shortest path costs
    optimal_node_lst = shortest_path[0]  # Grab the shortest path nodes from optimal path

    # For each edge pair in the optimal node list...
    for node in range(len(optimal_node_lst) - 1):

        # Stop if we reach the k limit
        if k == 0:
            break

        node_pair = (optimal_node_lst[node], optimal_node_lst[node + 1])  # Grab the current edge pair
        new_graph = copy.deepcopy(known_graph)  # Create a copy of the graph
        new_graph.weights[node_pair] = math.inf  # Set the current edge pair to infinity, such that it won't be selected
        k_path = dijkstra(new_graph, optimal_node_lst[0], optimal_node_lst[len(optimal_node_lst) - 1])  # Perform Dijkstra SP
        k_shortest.append(k_path[1])  # Return the cost of the shortest path found
        k -= 1  # Decrement the k-value

    # Sort costs by ascending order
    k_shortest.sort()

    # If we did not exhaust the k size
    if k != 0:

        # Pad the shortest path with the last node cost found
        final_k_path_node = k_shortest[len(k_shortest) - 1]
        while k:
            k_shortest.append(final_k_path_node)
            k -= 1

    return k_shortest


def ksp(graph, start, goal, k):
    """
    Calculate the K-Shortest Path
    :param graph: Weights and edges of nodes
    :param start: Starting node
    :param goal: Goal node to reach
    :param k: Amount of paths to find (assuming greater than 0)
    """
    k_shortest_paths = []

    # Perform Dijkstra shortest path search, adding it to the list of shortest paths
    dijkstra_sp = dijkstra(graph, start, goal)
    k_shortest_paths.append(dijkstra_sp[1])
    k -= 1

    # Perform alternate shortest path searches (for k - 1), adding them to the list of shortest paths
    alternate_sp = alt_shortest_paths(dijkstra_sp, graph, k)
    for route_cost in alternate_sp:
        k_shortest_paths.append(route_cost)

    return k_shortest_paths


def main():
    """
    Main of the program
    """
    try:
        abs_location = os.path.abspath(sys.argv[1])  # Location of input file

    except IndexError:
        print("Error: Program must be run from the commandline with arguments.")
        sys.exit(-1)

    try:
        input_f = open(abs_location, "r")

    except FileNotFoundError:
        print("Error: It appears that the input text file location (absolute location) was incorrect.")
        sys.exit(-1)

    print("Loading data... ")
    g = Graph()  # Declare our graph object

    # Input the node and edge counts
    no_of_nodes, no_of_edges = input_f.readline().split()

    no_of_edges = int(no_of_edges)
    no_of_nodes = int(no_of_nodes)

    # Input each of the edges and add them to the graph object
    for _ in range(no_of_edges):
        node_from, node_to, weight = input_f.readline().split()
        g.add_edge(node_from, node_to, float(weight))

    # Input the starting and goal node as well as k value
    start_node, goal_node, k_value, = input_f.readline().split()
    k_value = int(k_value)
    print("[Done]")
    print("Calculating k-shortest-paths... ")

    start_time = time.time()

    # Execute the k-shortest-paths algorithm
    found_path_costs = ksp(g, start_node, goal_node, k_value)

    end_time = time.time()

    print("[Done]")
    print("Found Shortest Path Distances in", "%0.2f (secs):" % (end_time - start_time))
    print(found_path_costs)

    input_f.close()


if __name__ == "__main__":
    main()
