#categorical cross entropy loss function for classification using softmax 

"""
sample loss value = -sum(target values(sample), output index * log(outputs(sample), output index))
                            ,       
            Li = -Eyi, jlog(yi, j)
                 j

one-hot encoding
    example:
    classes: 3
    label(index): 0
    one-hot: [1, 0, 0]

    example:
    classes: 4
    label: 2
    one-hot: [0, 0, 1, 0]

calculating categorical cross entropy
example:
    classes: 3
    label(index): 0
    one-hot: [1, 0, 0]
    prediction: [0.7, 0.1, 0.2] //from neural network
                            ,       
            Li = Eyi * jlog(yi, j)  =  -(1 * log(0.7) + 0 * log(0.1) + 0 * log(0.2))  =  0.35667...
                 j

"""

import numpy as np
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] + 
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

loss = -(math.log(softmax_output[0]) * target_output[0])

#print(loss)


softmax_outputs = np.array([
                          [0.7, 0.1, 0.2],
                          [0.1, 0.5, 0.4],
                          [0.02, 0.9, 0.08]
                          ])
class_targets = [0, 1, 1]
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
mean_loss = np.mean(neg_log)
print(mean_loss) #this will error if we have 0, then we're calling -log(0) and mean on that, big bad 