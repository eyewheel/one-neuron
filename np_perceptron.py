# with love to marsland (2009)

import numpy as np

def mcculloch_pitts_neuron(inputs, weights, threshold):
    return np.sum(np.dot(inputs, weights)) > threshold

class Perceptron:
    def __init__(self, size, threshold):
        self.weights = np.random.rand(size)
        self.threshold = threshold

    def activation(self, result):
        return result > self.threshold

    def classify(self, inputs):
        if inputs.shape != self.weights.shape:
            raise ValueError(f"inputs must be a 1d array of size {self.weights.shape[0]}")
        return self.activation(np.sum(np.dot(inputs, self.weights)))

p = Perceptron(2, 0)

print(p.classify(np.array([0, 1])))
