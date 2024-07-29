import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pulp
import time

# Improved Greedy Algorithm for Dominating Set
def improved_greedy_dominant(graph: nx.Graph) -> list:
    dominating_set = set()
    nodes = set(graph.nodes())
    
    while nodes:
        max_degree_node = max(nodes, key=lambda x: len(set(graph.neighbors(x)) & nodes))
        dominating_set.add(max_degree_node)
        nodes -= set(graph.neighbors(max_degree_node)) | {max_degree_node}
        
    return list(dominating_set)

# Minimum Dominating Set using Integer Linear Programming (ILP)
def ilp_dominating_set(graph: nx.Graph) -> list:
    model = pulp.LpProblem("Dominating_Set", pulp.LpMinimize)
    node_vars = {node: pulp.LpVariable(f'x_{node}', cat=pulp.LpBinary) for node in graph.nodes()}

    # Objective Function: Minimize the sum of the variables
    model += pulp.lpSum(node_vars.values())

    # Constraints: Ensure every node is either in the set or adjacent to a node in the set
    for node in graph.nodes():
        model += pulp.lpSum(node_vars[neighbor] for neighbor in graph.neighbors(node)) + node_vars[node] >= 1

    # Solve the problem
    model.solve()

    # Extract the solution
    dominating_set = [node for node in graph.nodes() if node_vars[node].value() == 1]
    
    return dominating_set

# Generate a random graph
def generate_random_graph(vertices: int, edges: int) -> nx.Graph:
    return nx.gnm_random_graph(vertices, edges)

# Performance Analysis
def performance_analysis():
    vertices = 250
    edges = 760
    G = generate_random_graph(vertices, edges)

    start_time = time.time()
    greedy_set = improved_greedy_dominant(G)
    greedy_time = time.time() - start_time

    start_time = time.time()
    ilp_set = ilp_dominating_set(G)
    ilp_time = time.time() - start_time

    print(f"Improved Greedy Dominant Set: {greedy_set}")
    print(f"Size of Improved Greedy Dominant Set: {len(greedy_set)}")
    print(f"Greedy Algorithm Time: {greedy_time:.4f} seconds")
    
    print(f"ILP Dominant Set: {ilp_set}")
    print(f"Size of ILP Dominant Set: {len(ilp_set)}")
    print(f"ILP Algorithm Time: {ilp_time:.4f} seconds")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    nx.draw(G, with_labels=True, node_color=['red' if node in greedy_set else 'blue' for node in G.nodes()])
    plt.title('Improved Greedy Algorithm')
    plt.subplot(122)
    nx.draw(G, with_labels=True, node_color=['red' if node in ilp_set else 'blue' for node in G.nodes()])
    plt.title('ILP Approach for Dominating Set')
    plt.show()

if __name__ == "__main__":
    performance_analysis()
