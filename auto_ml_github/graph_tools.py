import array
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import dgl


#Function takes in point cloud data and returns a networkx graph
def Points2graph(points=None, n_neighbours = None, algorithm = None):

    NN = NearestNeighbors(n_neighbors=n_neighbours, algorithm=algorithm).fit(points)
    adj_matrix = NN.kneighbors_graph(points).toarray()
    
    graph = nx.from_numpy_matrix(adj_matrix)
    
    return graph, adj_matrix



def Points2graph_dist(points = None, n_neighbours = None):

    # create the adjacency matrix based on the KNN graph
    num_points = points.shape[0]
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        distances = np.sum(((points - points[i]) ** 2), axis=-1)
        indices = np.argsort(distances)
        adj_matrix[i, indices[:n_neighbours]] = 1
    
    adj_matrix
    graph = nx.from_numpy_matrix(adj_matrix)
    
    return graph, adj_matrix


