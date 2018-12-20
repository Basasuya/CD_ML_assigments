import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    
    X2 = np.vstack((np.ones((1, X.shape[1])), X))
#     print(y.shape, X2.shape, w.shape)
#     step = 0.001
#     step2 = 0.001
#     iters = 0
#     while(1):
#         iters = iters + 1
#         w = w + step * np.matmul(X2, (y - np.matmul(X2.T, w).T).T) - step2 * w
        
#         error = np.sum(y != np.sign(np.matmul(X2.T, w).T))
#         if(iters > 100 or error == 0):
#             break
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X2, X2.T)), X2), y.T)

    
    return w
