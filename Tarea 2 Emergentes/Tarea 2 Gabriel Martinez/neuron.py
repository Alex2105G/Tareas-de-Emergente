import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        scale = np.sqrt(2.0 / max(n_inputs, 1))
        self.weights = np.random.randn(n_inputs) * scale
        self.bias = np.random.randn() * scale
        self.output = 0.0
        self.delta = 0.0
        self._last_input = None

    def activate(self, inputs):
        self._last_input = inputs
        z = np.dot(self.weights, inputs) + self.bias
        self.output = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return self.output

    def sigmoid_derivative(self):
        return self.output * (1.0 - self.output)
