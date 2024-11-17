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

class PerceptronLayer:
    def __init__(self, size, neurons, biases):
        # each column is a neuron
        self.weights = np.random.rand(size, neurons)
        print(self.weights)
        self.size = size
        # biases should be a vector (neurons,)
        if biases.shape[0] is not neurons:
            raise ValueError("must have 1 bias per neuron")
        self.biases = biases.reshape(-1, 1) # (1, neurons)
        self.learning_rate = 0.2

    def classify(self, inputs):
        # inputs should be a matrix of size (n, size)
        if (inputs.shape[1] != self.size):
            raise ValueError(f"input vectors must have dimension {size}")

        # this is a matrix where each row is the output for all the neurons
        # for a given input
        # these rows can then be compared to an output pattern
        raw_output = np.matmul(inputs, self.weights)
        activations = self.activation(raw_output)
        return activations

    def activation(self, results):
        return np.where(results > 0, 1, 0)

    def loss(self, expected, actual):
        return expected - actual

    def train(self, inputs, expected):
        # expected should be a vector of size (neurons,)
        # turn it into (neurons, 1)
        expected = expected.reshape(-1, 1)
        
        # subtracting a matrix from a vector
        # works because broadcasting turns expected into a matrix "stack"
        loss = self.loss(expected, self.classify(inputs))

        # should be, for each element of the loss matrix, 
        # multiply it by its relevant input vector and add it to its relevant
        # weight
        weights_update = self.learning_rate * np.matmul(np.transpose(inputs), loss)
        self.weights += weights_update 

p = PerceptronLayer(2, 3, np.zeros(3))



