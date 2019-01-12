import numpy as np
import sys
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    d = np.sum(W, axis=1)
    l= np.diag(d) - W
    s = np.diag(1.0 / (d ** (0.5)))
    result = np.dot(np.dot(s, l), s)
    lam, H = np.linalg.eig(result)
    F = np.take(H,lam.argsort()[:k], axis=-1)
    idx = kmeans(F, k)
    
    return idx
    # end answer
