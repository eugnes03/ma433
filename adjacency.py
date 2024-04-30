import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_adjacency_matrix(data):
    nodes = data['nodes']
    edges = data['edges']
    n = len(nodes)
    id_to_index = {node['id']: index for index, node in enumerate(nodes)}
    adjacency_matrix = np.zeros((n, n), dtype=int)

    for edge in edges:
        source_index = id_to_index[edge['source']]
        target_index = id_to_index[edge['target']]
        adjacency_matrix[source_index][target_index] = 1
        adjacency_matrix[target_index][source_index] = 1  # Assuming undirected graph
    return adjacency_matrix, id_to_index, n

def create_incidence_matrix(data, id_to_index, n):
    edges = data['edges']
    m = len(edges)
    incidence_matrix = np.zeros((n, m), dtype=int)
    for index, edge in enumerate(edges):
        source_index = id_to_index[edge['source']]
        target_index = id_to_index[edge['target']]
        incidence_matrix[source_index][index] = 1
        incidence_matrix[target_index][index] = -1  # Use -1 for directed graph; for undirected use 1
    return incidence_matrix

def create_laplacian_matrix(adjacency_matrix):
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix

def find_bottlenecks(laplacian_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    idx = eigenvalues.argsort()
    fiedler_vector = eigenvectors[:, idx[1]]
    return fiedler_vector

def visualize_bottlenecks(graph, fiedler_vector):
    pos = nx.spring_layout(graph)
    options = {
        "node_color": ['red' if val < 0 else 'blue' for val in fiedler_vector],
        "edge_color": 'gray',
        "node_size": 500,
        "width": 1.5,
        "with_labels": True,
        "font_weight": "bold",
        "font_color": "darkred"
    }
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, **options)
    plt.title('Graph Visualization with Bottlenecks Highlighted')
    plt.axis('off')
    plt.show()

def visualize_adjacency_matrix(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', width=2, font_size=15)
    plt.title('Enhanced Graph Visualization')
    plt.axis('off')
    plt.show()

# Load JSON data
json_file = 'friends.json'
with open(json_file, 'r') as file:
    data = json.load(file)

# Create matrices
adjacency_matrix, id_to_index, num_nodes = create_adjacency_matrix(data)
incidence_matrix = create_incidence_matrix(data, id_to_index, num_nodes)
laplacian_matrix = create_laplacian_matrix(adjacency_matrix)

# Find bottlenecks
fiedler_vector = find_bottlenecks(laplacian_matrix)

# Create graph from adjacency matrix
G = nx.from_numpy_array(adjacency_matrix)

# Print matrices and visualize
print("Adjacency Matrix:")
print(adjacency_matrix)
print("Incidence Matrix:")
print(incidence_matrix)
print("Laplacian Matrix:")
print(laplacian_matrix)
print("Fiedler Vector:")
print(fiedler_vector)

visualize_adjacency_matrix(adjacency_matrix)
visualize_bottlenecks(G, fiedler_vector)
