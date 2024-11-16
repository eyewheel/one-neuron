# with love to marsland (2009)

import numpy as np

def mcculloch_pitts_neuron(inputs, weights, threshold):
    return np.sum(np.dot(inputs, weights)) > threshold

class Perceptron:
    def __init__(self, size, threshold):
        self.weights = np.random.rand(size)
        self.threshold = threshold
        self.learning_rate = 0.2

    def activation(self, result):
        return int(result > self.threshold)

    def classify(self, inputs):
        if inputs.shape != self.weights.shape:
            raise ValueError(f"inputs must be a 1d array of size {self.weights.shape[0]}")
        return self.activation(np.sum(np.dot(inputs, self.weights)))

    def loss(self, expected, actual):
        return expected - actual

    def train(self, inputs, expected):
        actual = self.classify(inputs)
        self.weights += self.learning_rate * self.loss(expected, actual) * inputs
        return self.classify(inputs)

p = Perceptron(2, 0)

print(p.classify(np.array([0, 1])))
print(p.classify(np.array([1, 1])))
print(p.classify(np.array([0, 0])))
print(p.classify(np.array([1, 0])))
print("begin...")

while True:
    print(p.train(np.array([0, 1]), 0))
    print(p.train(np.array([1, 1]), 1))
    print(p.train(np.array([0, 0]), 0))
    print(p.train(np.array([1, 0]), 0))
    input(">")
