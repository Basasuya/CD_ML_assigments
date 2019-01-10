import numpy as np
import sys
def fullyconnect_feedforward(in_, weight, bias):
    '''
    The feedward process of fullyconnect
      input parameters:
          in_     : the intputs, shape: [number of images, number of inputs]
          weight  : the weight matrix, shape: [number of inputs, number of outputs]
          bias    : the bias, shape: [number of outputs, 1]

      output parameters:
          out     : the output of this layer, shape: [number of images, number of outputs]
    '''
    # TODO

    # begin answer
    new_in = np.hstack((in_, np.ones((in_.shape[0],1))))
    new_weight = np.vstack((weight, bias.T))
    out = np.matmul(new_in, new_weight)

    # end answer

    return out

