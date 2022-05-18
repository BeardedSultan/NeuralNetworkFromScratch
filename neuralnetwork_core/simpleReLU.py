"""
every neuron has an activation function after the inputs * weights + bias
##non linear activation functions
step functions - 0 or 1
sigmoid activation function - has more granularity than just 0 and 1
rectified linear unit activation funtion - linear when greater than 0 - super simple and fast calculation
    refer to pt.5 
granularity allows optimization
"""

import numpy as np

np.random.seed(0)

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
    ]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

#simple ReLU activation function
"""
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
"""
"""
for i in inputs:
    output.append(max(0, i))
"""

#ReLU object
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

reLU = Activation_ReLU()
reLU.forward(inputs)
print(reLU.output)

