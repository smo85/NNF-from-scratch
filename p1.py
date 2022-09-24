import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

# nnfs.init()

# X, y = spiral_data(100, 3)

# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
print(np.sum(norm_values, axis=1, keepdims=True))

'''
NOTES - Softmax
Trying to get a probability distribution
Can be used to determine how right or wrong a model is at that moment
'''
