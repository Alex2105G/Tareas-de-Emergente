import numpy as np


class Trainer:
    def __init__(self, mlp, learning_rate=0.1):
        self.mlp = mlp
        self.learning_rate = learning_rate

    def train(self, X, y, epochs):
        n_classes = self.mlp.n_outputs
        epoch_errors = []

        for epoch in range(epochs):
            total_error = 0.0
            indices = np.random.permutation(len(X))

            for i in indices:
                x_sample = X[i]
                target = np.zeros(n_classes)
                target[y[i]] = 1.0

                output = self.mlp.forward(x_sample)
                error = target - output
                total_error += float(np.sum(error ** 2)) / 2.0

                out_layer = self.mlp.layers[-1]
                for j, neuron in enumerate(out_layer.neurons):
                    neuron.delta = error[j] * neuron.sigmoid_derivative()

                for l in range(len(self.mlp.layers) - 2, -1, -1):
                    layer = self.mlp.layers[l]
                    next_layer = self.mlp.layers[l + 1]
                    for j, neuron in enumerate(layer.neurons):
                        grad = sum(
                            next_layer.neurons[k].weights[j] * next_layer.neurons[k].delta
                            for k in range(len(next_layer.neurons))
                        )
                        neuron.delta = grad * neuron.sigmoid_derivative()

                layer_input = x_sample
                for layer in self.mlp.layers:
                    for neuron in layer.neurons:
                        neuron.weights += self.learning_rate * neuron.delta * layer_input
                        neuron.bias += self.learning_rate * neuron.delta
                    layer_input = layer.get_outputs()

            epoch_errors.append(total_error / len(X))
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Época {epoch + 1}/{epochs}  error={epoch_errors[-1]:.6f}")

        self.mlp.total_epochs += epochs
        return epoch_errors
