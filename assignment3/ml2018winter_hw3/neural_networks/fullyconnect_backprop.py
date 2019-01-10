import numpy as np
import sys
def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape:   
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          out_sensitivity : the sensitivity to the lower layer, shape: 
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO

    # begin answer
    out_sensitivity = np.matmul(in_sensitivity, weight.T)
    new_in = np.hstack((in_, np.ones((in_.shape[0],1))))
    tmp_weight_grad = np.matmul(new_in.T, in_sensitivity) / in_.shape[0]
    weight_grad = tmp_weight_grad[:weight.shape[0],]
    bias_grad = tmp_weight_grad[weight.shape[0]:,].T
#     print(bias_grad.shape)
    # end answer

    return weight_grad, bias_grad, out_sensitivity

