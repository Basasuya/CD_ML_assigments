import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    X2 = np.vstack((np.ones((1, X.shape[1])), X))
    step = 0.01
    y = np.intc(y == 1) 
    iters = 0
    while(1):
        iters = iters + 1
        w = w + step * np.matmul(X2, (y - 1 / (1 + np.exp(-np.matmul(X2.T, w).T))).T ) + step * lmbda * w
        tmp = np.intc((1 / (1 + np.exp(-np.matmul(X2.T, w).T))) >= 0.5)
        error = np.sum(y != tmp)
        if(iters > 1000 or error == 0):
            break
#     w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X2, X2.T)), X2), y.T)

    
    return w
