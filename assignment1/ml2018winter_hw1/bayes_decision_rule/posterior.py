import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    
    p = np.zeros((C, N))
    #TODO

    # begin answer
    totalRow = np.sum(x, axis = 1)
    for i in range(C):
        sum = 0
        for j in range(N):
            p[i][j] = l[i][j] * totalRow[i] / total 
            sum += p[i][j]
    for j in range(N):
        sum = np.sum(p[:,j])
        for i in range(C):
            p[i][j] = p[i][j] / sum
    # end answer
    
    return p
