import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    M = X.shape[0]
    p = np.zeros((N, K))
    #Your code HERE
#     print(N, K, M)
    # begin answer
    for k in range(K):
        tmpSigma = np.matrix(Sigma[:,:,k])
        tmpMu = Mu[:,k]
        tmpPhi = Phi[k]
        tmpSigmaDet = np.linalg.det(tmpSigma)
        for i in range(N):
            x = X[:,i]
            p[i, k] = math.exp( -0.5* np.dot(np.dot( (x - tmpMu).T, tmpSigma.I), x - tmpMu ) / (2 * math.pi * math.sqrt(tmpSigmaDet) ) )
    for i in range(N):
        tmp = 0
        for k in range(K):
            p[i, k] = p[i, k] * Phi[k]
            tmp += p[i, k]
        for k in range(K):
            p[i, k] = p[i, k] / tmp
    # end answer
#     print(p)
    return p
    