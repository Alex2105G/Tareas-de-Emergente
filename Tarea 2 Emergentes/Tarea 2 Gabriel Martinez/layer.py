import numpy as np
from neuron import Neuron


class Layer:
    def __init__(self, n_neurons, n_inputs, layer_type='h'):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]
        self.layer_type = layer_type
        self.inputs = None

    def forward(self, inputs):
        self.inputs = np.array(inputs, dtype=float)
        return np.array([n.activate(self.inputs) for n in self.neurons])

    def get_outputs(self):
        return np.array([n.output for n in self.neurons])
