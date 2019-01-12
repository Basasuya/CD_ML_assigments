import numpy as np
import matplotlib.pyplot as plt
import sys
from pca import PCA
def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''

    # YOUR CODE HERE
    # begin answer
    img_r = (plt.imread(filename)).astype(np.float64)
    data = []
    for i in range(img_r.shape[0]):
        for j in range(img_r.shape[1]):
            if(img_r[i][j][3] > 0):
                tt = (img_r[i][j][0] * 299 + img_r[i][j][1]*587 + img_r[i][j][2]*114 + 500) / 1000
                if(tt > 150):
                    data.append([i, j])
    data = np.array(data)

    w, _ = PCA(data)
    result = np.dot(data, w)
    result_min = np.min(result, axis = 0)
    result_max = np.max(result, axis = 0)
    result -= result_min
    result_max = np.max(result, axis = 0)
    result[:,0] = (result[:,0] / result_max[0]) * 99
    result[:,1] = (result[:,1] / result_max[1]) * 99
    result_max = np.max(result, axis = 0)
    new_img_r = np.zeros((100, 100, 3))
    for i in range(result.shape[0]):
        new_img_r[-int(result[i][1])][-int(result[i][0])] = [100,100,100]  
    return new_img_r

    # end answer