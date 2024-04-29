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


def laplacian_eigenvalues(laplacian_matrix):
    # Compute the eigenvalues of the Laplacian matrix
    eigenvalues = np.linalg.eigvals(laplacian_matrix)
    return np.sort(eigenvalues)  # Returning sorted eigenvalues


def visualize_adjacency_matrix(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(G)  # using a different layout

    # Node customization
    node_colors = ['skyblue' if degree > 1 else 'lightgreen' for _, degree in G.degree()]
    node_sizes = [700 if degree > 1 else 1000 for _, degree in G.degree()]

    # Edge customization
    edge_colors = ['gray' if weight == 1 else 'purple' for _, _, weight in G.edges(data='weight', default=1)]
    edge_widths = [2 if weight == 1 else 4 for _, _, weight in G.edges(data='weight', default=1)]

    # Draw the network
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, edge_color=edge_colors, width=edge_widths,
            with_labels=True, font_size=15, font_color='darkred', font_weight='bold', alpha=0.9)
    
    plt.title('Enhanced Graph Visualization', size=20, color='darkblue')
    plt.axis('off')  # Hide the axes
    plt.show()

# Load JSON data
json_file = 'friends.json'
with open(json_file, 'r') as file:
    data = json.load(file)

# Create matrices
adjacency_matrix, id_to_index, num_nodes = create_adjacency_matrix(data)
incidence_matrix = create_incidence_matrix(data, id_to_index, num_nodes)
laplacian_matrix = create_laplacian_matrix(adjacency_matrix)
eigenvalues = laplacian_eigenvalues(laplacian_matrix)

# Print matrices
print("Adjacency Matrix:")
print(adjacency_matrix)
print("Incidence Matrix:")
print(incidence_matrix)
print("Laplacian Matrix:")
print(laplacian_matrix)
print(eigenvalues)

visualize_adjacency_matrix(adjacency_matrix)
