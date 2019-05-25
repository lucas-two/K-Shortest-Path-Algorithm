"""
K-SHORTEST-PATH PROBLEM
Author: Lucas Geurtjens
Date: 25/05/2019

REFERENCES/CREDITS:
[1]
Dijkstra's SP implementation and graph structure was heavily influenced by 'Ben Alex Keen'
URL : http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/

[2]
Yen's algorithm...
"""

from collections import defaultdict


class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Assuming non bi-directional
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight


def dijkstra(graph, start, goal):
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

    optimal_path = optimal_path[::-1]  # Reverse the path (since we went from end -> start

    return optimal_path, optimal_cost


g = Graph()

edges = [
    ("C", "D", 3),
    ("C", "E", 2),
    ("D", "F", 4),
    ("E", "D", 1),
    ("E", "F", 2),
    ("E", "G", 3),
    ("F", "G", 2),
    ("F", "H", 1),
    ("G", "H", 2)
]


for edge in edges:
    g.add_edge(*edge)

my_path = dijkstra(g, 'C', 'H')
print(my_path)
