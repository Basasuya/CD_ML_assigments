import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of  fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    data = data - np.mean(data, axis=0, keepdims=True)
    c = np.cov(data.T)
    eigvalue, eigvector = np.linalg.eig(c)
    eigvalue_argsort = np.argsort(eigvalue)[::-1]
    eigvalue = eigvalue[eigvalue_argsort]
    eigvector = eigvector[:, eigvalue_argsort]
    return eigvector.astype(float), eigvalue.astype(float)
    # end answer