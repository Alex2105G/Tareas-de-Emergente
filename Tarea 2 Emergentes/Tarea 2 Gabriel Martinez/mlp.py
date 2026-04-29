import numpy as np
from layer import Layer


class MLP:
    def __init__(self, n_inputs, n_outputs, n_hidden_layers, n_neurons_per_layer):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.total_epochs = 0
        self.layers = []
        self._build()

    def _build(self):
        self.layers = []
        prev = self.n_inputs
        for _ in range(self.n_hidden_layers):
            self.layers.append(Layer(self.n_neurons_per_layer, prev, 'h'))
            prev = self.n_neurons_per_layer
        self.layers.append(Layer(self.n_outputs, prev, 'o'))

    def forward(self, inputs):
        current = np.array(inputs, dtype=float)
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def predict(self, inputs):
        out = np.array(self.forward(inputs), dtype=float)
        if self.n_outputs == 1:
            return int(out.flat[0] >= 0.5)
        return int(np.argmax(out))

    def predict_all(self, X):
        return [self.predict(x) for x in X]
