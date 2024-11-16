# with love to marsland (2009)

import numpy as np

def mcculloch_pitts_neuron(inputs, weights, threshold):
    return np.sum(np.dot(inputs, weights)) > threshold

class Perceptron:
    def __init__(self, size):
        self.weights = np.random.rand(size)

p = Perceptron()
print(p.weights)

