import numpy as np
import sys
from scipy.spatial import distance

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
#     center = np.random.randint(low=0, high=x.shape[0], size=None, dtype='l')
    center = []
    hashs = {}
    
    for i in range(k):
        tt = 0
        while(True):
            tt = np.random.randint(0, high=x.shape[0], size=1)[0]
            if(tt in hashs):
                continue
            else:
                hashs[tt] = 1
                break
        center.append(x[tt])
#     sys.exit()
    center = np.array(center)
    idx = np.zeros(x.shape[0]).astype(np.int32)
    iter_ctrs = []
    
    iterssss = 0
    for iters in range(10):
        iterssss = iterssss + 1
        cnt = 0
        sum2 = np.zeros(k)
        sum1 = np.zeros((k, x.shape[1]))
        dis = distance.cdist(x, center, 'euclidean')
        idx = np.argmin(dis, axis = 1)

        for i in range(idx.shape[0]):
            sum1[idx[i]] = sum1[idx[i]] + x[i]
            sum2[idx[i]] = sum2[idx[i]] + 1
        for i in range(center.shape[0]):
            center[i] = sum1[i] / sum2[i]
        iter_ctrs.append(center)
#         print(sum2)
#         if(cnt == 0 and iters > 100):
#             break
    iter_ctrs = np.array(iter_ctrs);
    # end answer
    return idx, center, iter_ctrs
