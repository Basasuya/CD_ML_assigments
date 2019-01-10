import numpy as np
import scipy.stats
import sys
from scipy.spatial import distance
def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

#     y = np.zeros(x.shape[0])
#     for i in range(x.shape[0]):
#         tmp_x = x[i,:]
#         tt = np.sum((x_train - tmp_x) * (x_train - tmp_x), axis = 1)
#         result , _ = scipy.stats.mode(y_train[np.argsort(tt)[:k]])
#         y[i] = result
        

  
    # begin answer
    dis = distance.cdist(x, x_train, 'euclidean')
    y , _ = scipy.stats.mode(y_train[np.argsort(dis)[:, :k]], axis = 1)
    # end answer

    return y
