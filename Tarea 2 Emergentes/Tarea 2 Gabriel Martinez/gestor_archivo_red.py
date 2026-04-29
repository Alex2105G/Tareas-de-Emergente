"""
Persistencia de la MLP en CSV según el enunciado (Tarea 2 PDF).

- Fila 1: encabezado de la tabla.
- Filas 2–6: hiperparámetros u, v, L, b, e (col. 1 = letra, col. 2 = valor;
  columnas 3 en adelante vacías hasta completar el ancho de la tabla).
- Desde la fila 7: una fila por neurona — tipo, capa, posición, luego pesos
  w0, w1, … donde w0 es el peso de la entrada 0 (sesgo) y el resto coinciden
  con los pesos sinápticos de la neurona.

c del PDF para columnas de peso: se usa ancho w_cols = max(u, b) + 1 para
cubrir fan-in u+1 y b+1 en todas las capas (coherente con c = max(u, b) del
texto si se interpreta el ancho mínimo necesario para filas de pesos).
"""
import csv
import numpy as np
from mlp import MLP


class GestorArchivoRed:
    @staticmethod
    def _row_width(u, b):
        """Ancho uniforme: 3 etiquetas + columnas de peso (c >= max(u,b) del PDF; +1 para fan-in)."""
        return 3 + max(u, b) + 1

    def save(self, mlp, filepath):
        u = mlp.n_inputs
        b = mlp.n_neurons_per_layer
        width = self._row_width(u, b)
        n_w = width - 3
        headers = ['tipo', 'capa', 'posicion'] + [f'w{i}' for i in range(n_w)]
        rows = [headers]

        for key, val in [
            ('u', u),
            ('v', mlp.n_outputs),
            ('L', mlp.n_hidden_layers),
            ('b', b),
            ('e', mlp.total_epochs),
        ]:
            # Cols. 3 en adelante vacías (misma anchura que filas de neuronas).
            rows.append([key, val] + [''] * (width - 2))

        for layer_idx, layer in enumerate(mlp.layers):
            layer_num = layer_idx + 1
            for pos, neuron in enumerate(layer.neurons):
                full = [float(neuron.bias)] + [float(w) for w in neuron.weights]
                padded = full + [0.0] * (n_w - len(full))
                rows.append(
                    [layer.layer_type, layer_num, pos] + padded[:n_w]
                )

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def load(self, filepath):
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_rows = list(reader)

        header = all_rows[0]
        legacy_sesgo = len(header) > 0 and header[-1].strip().lower() == 'sesgo'

        params = {}
        for row in all_rows[1:6]:
            params[row[0].strip()] = int(float(row[1]))

        u = params['u']
        v = params['v']
        L = params['L']
        b = params['b']
        e = params['e']

        mlp = MLP(u, v, L, b)
        mlp.total_epochs = e

        for row in all_rows[6:]:
            if not row or not row[0].strip():
                continue
            ntype = row[0].strip()
            layer_num = int(row[1])
            pos = int(row[2])
            if ntype == 'h':
                layer_idx = layer_num - 1
            else:
                layer_idx = len(mlp.layers) - 1

            layer = mlp.layers[layer_idx]
            neuron = layer.neurons[pos]
            n_in = len(neuron.weights)

            if legacy_sesgo:
                neuron.weights = np.array([float(row[3 + i]) for i in range(n_in)])
                neuron.bias = float(row[-1])
            else:
                neuron.bias = float(row[3])
                neuron.weights = np.array([float(row[4 + i]) for i in range(n_in)])

        return mlp
