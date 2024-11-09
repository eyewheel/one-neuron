import numpy as np

# basically working through https://neuralnetworksanddeeplearning.com/chap1.html

def perceptron(inputs, weights):
    if np.dot(inputs, weights) > 0:
        return 1
    else:
        return 0

# -1 weighted 5 + 1 weighted 1
# print(perceptron([-1, 1], [5, 1]))

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
# print(sigmoid_neuron([-1, 1], [5, 1], 0))

# vector of inputs
# vector of weights for each input, for each neuron
# so 2d matrix
# put bias as the last "weight", is typical
# for each set of weights, run sigmoid_neuron on dot with inputs
# the outputs from each set of weights form the next vector inputs

input_dim = 4
inputs = np.random.rand(input_dim)

neuron_count = 2
# each row in this 2d matrix is a neuron
weights = np.random.rand(neuron_count, input_dim)

# save bias for later, still works with bias 0

def np_sigmoid(z):
    return 1 / (1 + np.exp(-z))

# we want to dot product each row of weights with the inputs
# and make a new vector with size neuron_count
# and then run the sigmoid element wise on dots

dots = np.matmul(weights, inputs)

outputs = np_sigmoid(dots)

print(outputs)

