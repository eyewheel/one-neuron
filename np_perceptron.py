# with love to marsland (2009)

import numpy as np

def mcculloch_pitts_neuron(inputs, weights, threshold):
    return np.sum(np.dot(inputs, weights)) > threshold

class PerceptronNeuron:
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

class Perceptron:
    def __init__(self, size, neurons, threshold):
        self.weights = np.random.rand(neurons, size)
        self.threshold = threshold
        self.learning_rate = 0.2

    def classify(self, inputs):
        # run activation on the dot product of the inputs with each neuron
        return self.activation(np.matmul(self.weights, inputs))

    def activation(self, results):
        return np.where(results > self.threshold, 1, 0)

    def loss(self, expected, actual):
        return expected - actual

    def train(self, inputs, expected):
        actual = self.classify(inputs)

        # currently we have two vectors of different sizes - the input
        # will be of size (size,), while the loss will be of size (neurons,)
        # (1, size)
        inputs = inputs.reshape(1, -1)
        # (neurons, 1)
        loss = self.loss(expected, actual).reshape(-1, 1)
        # numpy broadcasting will create a matrix where each row is the
        # loss across inputs for a given neuron.
        # each row in the weights is a neuron, so we can simply add
        update_weights = (loss * inputs)
        self.weights += update_weights

p = Perceptron(2, 3, 0)

print(p.classify(np.array([0, 1])))
print(p.classify(np.array([1, 1])))
print(p.classify(np.array([0, 0])))
print(p.classify(np.array([1, 0])))
print("begin...")

while True:
    p.train(np.array([0, 1]), 1)
    p.train(np.array([1, 1]), 1)
    p.train(np.array([0, 0]), 0)
    p.train(np.array([1, 0]), 0)

    print(p.classify(np.array([0, 1])))
    print(p.classify(np.array([1, 1])))
    print(p.classify(np.array([0, 0])))
    print(p.classify(np.array([1, 0])))
    input(">")
