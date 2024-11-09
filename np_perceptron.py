import numpy as np

# basically working through https://neuralnetworksanddeeplearning.com/chap1.html

def perceptron(inputs, weights):
    if np.dot(inputs, weights) > 0:
        return 1
    else:
        return 0

# -1 weighted 5 + 1 weighted 1
print(perceptron([-1, 1], [5, 1]))

# can be called by various types of threshold
def calc_weighted(inputs, weights):
    return np.dot(inputs, weights)

"""
def perceptron(inputs, weights):
    if calc_weighted(inputs, weights) > 0:
        return 1
    else:
        return 0
"
"""

import math

def sigmoid_neuron(inputs, weights, bias):
    return (1 / (1 + math.exp(-(calc_weighted(inputs,weights) + bias))))

# with bias 0, should return values < or > 0 identically to perceptron
print(sigmoid_neuron([-1, 1], [5, 1], 0))







