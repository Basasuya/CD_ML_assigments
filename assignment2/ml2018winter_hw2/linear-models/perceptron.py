import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    X2 = np.vstack((np.ones((1, X.shape[1])), X))
    while(1):
        iters = iters + 1
        y2 = np.sign(np.matmul(w.T, X2))
        cnt = 0
        for i in range(y.shape[1]):
            if(y2[0, i] != y[0, i]):
                for j in range(len(w)):
                    w[j, 0] = w[j, 0] + X2[j, i] * y[0, i]
#                 w = w + X2 * y
                cnt = cnt + 1
        if(cnt == 0 or iters > 1000):
            break
    return w, iters