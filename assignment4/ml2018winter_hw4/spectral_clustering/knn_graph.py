import numpy as np
from scipy.spatial import distance
import sys

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    k = max(X.shape[0], k)
    dis = distance.cdist(X, X, 'euclidean')
    dis[np.argsort(dis)[:,k:]] = 0
    dis[dis > threshold] = 0
    return dis
    # end answer
