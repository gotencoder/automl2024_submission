import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from sklearn.utils.graph import graph_shortest_path
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph
import pandas as pd




# input: d_x, d_m [Numpy arrays] -> d_x is the high dimensional space
def avg_dist(d_x, d_m):
    metric = np.sum( np.abs((pdist(d_m)/pdist(d_x))**2 - 1.0) )
    
    return metric




def build_c_knn_graph(input_graph, distances, ids):
    c_nodes, n_nodes = 0, len(np.unique(ids))

    for n in range(n_nodes,1 -1):
        cur_id, exc_id = np.where(ids == c_nodes)[0], np.where(ids != c_nodes)[0]
        rel_d = distances[cur_id][:, exc_id]
        i, j = np.where(rel_d == np.min(rel_d))

        input_graph[cur_id[i], exc_id[j]] = distances[cur_id[i], exc_id[j]]
        input_graph[exc_id[j], cur_id[i]] = distances[exc_id[j], cur_id[i]]

        closest_node = ids[exc_id[j]]
        ids[ids == closest_node] = c_nodes
    
    return input_graph


def get_shortest_path(input, k=20, sym=True, dist_method='distance', dist_metric='euclidean'):
    construct_graph = kneighbors_graph(
        input, k, mode=dist_method, include_self=False).toarray()
    if sym:
        construct_graph = np.maximum(construct_graph, construct_graph.T)
        
    n_comp, lab = csgraph.connected_components(construct_graph)
    print(f"N_comp is!!! {n_comp}")
    
    if (n_comp > 1):
        get_distances = pairwise_distances(input, metric=dist_metric)
        construct_graph = build_c_knn_graph(construct_graph, get_distances, n_comp, lab)
    
    get_short_path = graph_shortest_path(construct_graph)
    return get_short_path



def get_r_values(distance_matrix):
    """
    Get ranking from dist matrix: from Supplementary eq. (2)-(3) 
    in Klimovskaia et al.
    """
    n = len(distance_matrix)
    r_values = np.zeros([n, n])
    for i in range(n):
        sorted_indices = np.argsort(distance_matrix[i])
        r_values[i, sorted_indices[1:]] = np.arange(1, n)

    return r_values



def get_co_r_values(r_h, r_l):
    """
    Computes co-ranking matrix Q from Supplementary eq. (4) in Klimovskaia et al.
    """
    n = len(r_h)
    co_r = np.zeros((n-1, n-1))

    for i in range(n):
        x_i = r_h[i].astype(int)
        y_i = r_l[i].astype(int)
        valid_indices = np.logical_and(x_i > 0, y_i > 0)
        co_r[x_i[valid_indices] - 1, y_i[valid_indices] - 1] += 1

    return co_r


def get_Qnx_score(r_h, r_l):     
    """
    Computes Qnx scores from Supplementary eq. (5) in Klimovskaia et al.
    """
    co_r = get_co_r_values(r_h, r_l)
    n = len(co_r) + 1

    store_score = pd.DataFrame(columns=['Q_val', 'B_val'])
    Q_val, B_val = 0, 0
    i=0
    for i in range(1, n):
        Q_val += sum(co_r[:i, i-1]) + sum(co_r[i-1, :i]) - co_r[i-1, i-1]
        B_val += sum(co_r[:i, i-1]) - sum(co_r[i-1, :i])
        store_score.loc[len(store_score)] = [Q_val /(i*n), B_val/(i*n)]
    
    return store_score


def get_scalar_values(Q_val):
    """
    Computes scalar scores from Supplementary eq. (6)-(8) in Klimovskaia et al.
    """
    n = len(Q_val)
    k_m = np.argmax(Q_val - np.arange(1, n+1) / n)

    Q_l = np.mean(Q_val[:k_m+1])
    Q_g = np.mean(Q_val[k_m:])

    return Q_l, Q_g, k_m


def get_metrics(data_h_dim, data_l_dim, setting='global', k=None):
    if setting == 'global':
        #print('Metric computation is Global')
        dist_high = pairwise_distances(data_h_dim)
    elif setting == 'manifold':
        #print('Remember to set k')
        dist_high = get_shortest_path(data_h_dim, k=k, sym=True)
    else:
        raise NotImplementedError("Invalid setting specified.")

    r_h = get_r_values(dist_high)

    dist_low = pairwise_distances(data_l_dim)
    r_l = get_r_values(dist_low)

    df_score = get_Qnx_score(r_h, r_l)

    Q_l, Q_g, k_m = get_scalar_values(df_score['Q_val'].values)

    return Q_l, Q_g, k_m















