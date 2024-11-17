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
        self.weights = np.random.rand(size + 1, neurons)
        # biases should be a vector (neurons,)
        if biases.shape[0] is not neurons:
            raise ValueError("must have 1 bias per neuron")
        self.biases = biases.reshape(-1, 1) # (1, neurons)
        self.learning_rate = 0.2

    def classify(self, inputs):
        inputs = self.with_bias(inputs)
        print(inputs)
        # run activation on the dot product of the inputs with each neuron
        return self.activation(np.matmul(self.weights, inputs))

    def with_bias(self, inputs):
        # turn inputs into a matrix stack of itself, one input per neuron
        broadcaster = np.ones((self.weights.shape[0], 1))
        inputs = broadcaster * inputs
        # append biases as the last column
        inputs = np.hstack((inputs, self.biases))
        return inputs

    def activation(self, results):
        return np.where(results > 0, 1, 0)

    def loss(self, expected, actual):
        return expected - actual

    def train(self, inputs, expected):
        actual = self.classify(inputs)
        inputs = self.with_bias(inputs)
        loss = self.loss(expected, actual)
        # numpy broadcasting will create a matrix where each row is the
        # loss across inputs for a given neuron.
        update_weights = (loss * inputs)
        self.weights += update_weights

p = PerceptronLayer(2, 3, np.zeros(3))

print(p.classify(np.array([1, 1])))


