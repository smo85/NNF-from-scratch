import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)  # structured this way so we don't have to transpose the weights
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
     self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2,5)
activation = Activation_ReLU()

layer1.forward(X)
activation.forward(layer1.output)
print(activation.output)




'''
NOTES - Activation functions
2+ hidden layers to fit non-linear
why - linear can only do linear problems so need non-linear (ReLU is close to linear, but the less than 0 makes it non-linear)
sigmoid (between 0 and 1) - has vanishing gradient problem
ReLU - (between 0 and whatever x is) - no vanishing gradient problem
    1. faster than sigmoid and simpler
    2. works well
'''
