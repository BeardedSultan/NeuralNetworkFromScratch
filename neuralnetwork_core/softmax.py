import math
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

#RAW_PYTHON_IMPLEMENTATION#
#exponentiation solves the negative number issue in ReLU, and leaves numbers in relative scale - e^x
'''E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

#next step is to normalize values
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)


#NUMPY_IMPLEMENTATION#
#exponentiation
exp_values = np.exp(layer_outputs)
#normalize
norm_values = exp_values / np.sum(exp_values)'''


#exponentiation + normalization = softmax activation function


#BATCH_IMPLEMENTATION#
layer_outputs_batch = [
                      [4.8, 1.21, 2.385],
                      [8.9, -1.81, 0.2],
                      [1.41, 1.051, 0.026]
                      ]

exp_values = np.exp(layer_outputs_batch)
sum = np.sum(layer_outputs_batch, axis=1, keepdims=True) #axis 1 is the sum of the rows, 0 is the columns
                                                         #keepdims keeps dimensions of the matrix
norm_values = exp_values / sum


#overflow is a problem with exponentiation
#overflow prevention by subtracting largest value from all values in layer prior to exponentiation