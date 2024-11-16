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

# the classic intuition for the dot product is the similarity between two
# vectors. this still applies here, as the inputs are "similar" to the weights
# when large-valued inputs have large-valued weights, creating a larger
# similarity score.

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

def run_layer(inputs, weights):
    if (weights.shape[1] != inputs.shape[0]):
        raise "weights must have same width as inputs"
    raw_outputs = np.matmul(weights, inputs)
    return np_sigmoid(raw_outputs)

run_layer(inputs, weights)

# when making the perceptron multi-layered,
# we want to feed outputs to inputs. each layer can have a different number
# of neurons; the neurons of each layer should have 
# (current_layer - 1).neuron_count inputs
# Let's define the layers as a vector where layers[0] returns the number of
# neurons in that layer
# for each layer, instantiate N neurons with previous_layer.N inputs

layers = np.array([2, 3, 2])

# wait, it would have to be a jagged array

# note it still learns nothing
