import numpy as np

def sigmoid(x):
    #Sigmoid f(x) = 1 / (1 + e^-x)
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.hidden1 = Neuron(weights=weights, bias=bias)
        self.hidden2 = Neuron(weights=weights, bias=bias)
        self.output1 = Neuron(weights=weights, bias=bias)

    def feedforward(self, x):
        output_hidden1 = self.hidden1.feedforward(x)
        output_hidden2 = self.hidden2.feedforward(x)

        output_output1 = self.output1.feedforward(np.array([output_hidden1, output_hidden2]))

        return output_output1
    
n = NeuralNetwork()
x = np.array([2, 3])
ans = n.feedforward(x)
print(ans)
