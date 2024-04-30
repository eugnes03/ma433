import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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

def spectral_clustering(laplacian_matrix, num_clusters):
    # Step 2: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    
    # Step 3: Sort the eigenvalues and select the smallest k eigenvectors (excluding the zero eigenvalue)
    idx = np.argsort(eigenvalues)[1:num_clusters+1]  # 1:num_clusters+1 to exclude the first eigenvector
    selected_eigenvectors = eigenvectors[:, idx]
    
    # Step 4: Normalize the rows to have unit length
    U = selected_eigenvectors
    norm = np.linalg.norm(U, axis=1)
    U_normalized = U / norm[:, np.newaxis]
    
    # Step 5: Cluster the points using K-means on the rows of U
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(U_normalized)
    labels = kmeans.labels_
    
    return labels

def visualize_clusters(graph, labels):
    # Map the labels to colors for visualization
    color_map = plt.get_cmap('viridis', np.unique(labels).size)
    node_colors = [color_map(labels[i]) for i in range(len(labels))]

    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, node_size=700, edge_color='gray', width=2, font_size=15)
    plt.title('Graph with Clustered Nodes')
    plt.axis('off')
    plt.show()


def check_connectivity(laplacian_matrix):
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(laplacian_matrix)
    # Count the number of zero eigenvalues which indicate the number of connected components
    num_connected_components = np.sum(np.isclose(eigenvalues, 0))
    return num_connected_components, eigenvalues

def spectral_clustering(laplacian_matrix, num_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    idx = np.argsort(eigenvalues)[1:num_clusters+1]  # Skip the first zero eigenvalue
    selected_eigenvectors = eigenvectors[:, idx]
    U_normalized = selected_eigenvectors / np.linalg.norm(selected_eigenvectors, axis=1)[:, np.newaxis]
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(U_normalized)
    return kmeans.labels_

def visualize_clusters(graph, labels):
    pos = nx.spring_layout(graph)
    color_map = plt.get_cmap('viridis', np.unique(labels).size)
    node_colors = [color_map(labels[i]) for i in range(graph.number_of_nodes())]
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, node_size=500, edge_color='gray', width=2, font_size=15)
    plt.title('Graph with Clustered Nodes')
    plt.axis('off')
    plt.show()


def is_valid_coloring(graph, colors):
    for node in range(graph.number_of_nodes()):
        for neighbor in graph.neighbors(node):
            if colors[node] == colors[neighbor] and colors[node] != -1:
                return False
    return True

def chromatic_number_util(graph, colors, num_colors, node=0):
    if node == graph.number_of_nodes():
        return is_valid_coloring(graph, colors)

    for color in range(num_colors):
        colors[node] = color
        if chromatic_number_util(graph, colors, num_colors, node + 1):
            return True
        colors[node] = -1  # Reset the color before backtracking
    return False

def find_chromatic_number(graph):
    num_nodes = graph.number_of_nodes()
    colors = [-1] * num_nodes

    for num_colors in range(1, num_nodes + 1):
        if chromatic_number_util(graph, colors, num_colors, 0):
            return num_colors  # Return the minimum number of colors needed

def visualize_colored_graph(graph, colors):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, node_color=colors, with_labels=True, node_size=500, cmap=plt.cm.viridis, edge_color='gray', width=2)
    plt.title('Graph Coloring')
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



num_clusters = 4  # Number of clusters to form
labels = spectral_clustering(laplacian_matrix, num_clusters)
visualize_clusters(G, labels)


num_connected_components, _ = check_connectivity(laplacian_matrix)
print(f"Number of Connected Components: {num_connected_components}")

labels = spectral_clustering(laplacian_matrix, num_connected_components)

visualize_clusters(G, labels)

#chromatic_num = find_chromatic_number(G)
#colors = [-1] * G.number_of_nodes()
#chromatic_number_util(G, colors, chromatic_num)
#visualize_colored_graph(G, colors)

matrixtest = [
[0,1,1,1,0,0,0,0],
[1,1,0,1,0,1,0,0],
[1,1,0,1,0,1,0,0],
[1,0,1,0,1,1,1,0],
[0,1,0,1,0,0,1,1],
[0,0,1,1,0,0,1,1],
[0,0,0,1,1,1,0,1],
[0,0,0,0,1,1,1,0]
]
eigenvalues1, eigenvectors1 = np.linalg.eigh(matrixtest)

print(eigenvalues1)