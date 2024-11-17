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
        self.size = size
        # biases should be a vector (neurons,)
        if biases.shape[0] !=  neurons:
            raise ValueError("must have 1 bias per neuron")
        self.biases = biases # keep 1d for broadcasting
        self.learning_rate = 0.1

    def classify(self, inputs):
        # inputs should be a matrix of size (N, size)
        if (inputs.shape[1] != self.size):
            raise ValueError(f"input vectors must have dimension {self.size}")

        # this is a matrix where each row is the output for all the neurons
        # for a given input
        # these rows can then be compared to an output pattern
        raw_output = np.matmul(inputs, self.weights) + self.biases
        activations = self.activation(raw_output)
        return activations

    def activation(self, results):
        return np.where(results > 0, 1, 0)

    def loss(self, expected, actual):
        return expected - actual

    def train(self, inputs, expected):
        # expected should be a vector of size (N, neurons)
        
        # subtracting a matrix from a vector
        # works because broadcasting turns expected into a matrix "stack"
        loss = self.loss(expected, self.classify(inputs))

        # should be, for each element of the loss matrix, 
        # multiply it by its relevant input vector and add it to its relevant
        # weight
        weights_update = self.learning_rate * np.matmul(np.transpose(inputs), loss)
        self.weights += weights_update 
        
        # update biases
        self.biases += self.learning_rate * np.sum(loss, axis=0)

p = PerceptronLayer(2, 3, np.ones(3))

print(p.classify(np.array([[0, 1]])))

while True:
    # neurons run AND, OR, and FIRST respectively
    p.train(np.array([
        [0, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [1, 1],
        [0, 1],
        [0, 1]
        ]),
        np.transpose(np.array([
            [0, 1, 0, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 0]
         ]))
    )

    print(p.classify(np.array([[0, 1]])))
    input(">")
