# one handwritten neuron ^_^

import random

class Neuron:
    def __init__(self, size, lr):
       self.size = size
       self.weights = []
       self.lr = lr # learning rate

       # fill it with the default weight (small random value)
       for i in range(size):
           self.weights.append(self.default_weight())

    def default_weight(self):
        return (random.random() * 0.1) - 0.05

    def recall(self, input):
        if len(input) != self.size - 1:
            raise Exception("This input is not appropriately sized for this neuron; please provide a %s-dimensional vector"%(self.size - 1))

        # Convention is for the bias to be at x_0, so we place it at the beginning of the list
        input.insert(0, self.bias())

        s = 0 # sum
        for i in range(self.size):
            s += (self.weights[i] * input[i])
        activation = self.activation(s)
        print("Returned a value of %s"%(activation))
        return activation

     # One round of supervised learning on one input vector.
    def train(self, input, target):
        activation = self.recall(input)
        print("Target was %s"%(target))
        for i in range(self.size):
            self.weights[i] += self.lr * (target - activation) * input[i]
        return activation

    # Currently simply a threshold function
    def activation(self, sum):
        if sum <= 0:
            return 0
        else:
            return 1

    def bias(self):
        return -1

    def __str__(self):
        return str(self.weights)

n = Neuron(4, 0.25)
print(n)
for i in range(30):
    n.train([0, 1, 2], 1)
    n.train([1, 2, 3], 1)
    n.train([1, 0, 2], 0)
    n.train([1, 3, 2], 0)
    n.train([2, 3, 4], 1)
    n.train([4, 1, 0], 0)
    n.train([4, 2, 3], 0)
    print(n)

# Test on data not in dataset
n.recall([3, 4, 5])
n.recall([44, 45, 46])
