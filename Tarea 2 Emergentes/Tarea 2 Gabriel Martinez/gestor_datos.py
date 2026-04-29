"""
Carga y validación de CSV (misma idea que gestor_datos de la Tarea 1):
última columna = clase; las demás = entradas.
"""
import numpy as np
import pandas as pd


class GestorDatos:
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.class_to_idx = None
        self.idx_to_class = None

    def load(self, filepath):
        df = pd.read_csv(filepath)
        self.X = df.iloc[:, :-1].values.astype(float)
        raw_labels = df.iloc[:, -1].values
        self.classes = sorted(set(raw_labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.y = np.array([self.class_to_idx[c] for c in raw_labels])
        return self.X, self.y

    def validate_dimensions(self, n_inputs, n_outputs):
        if self.X is None:
            return False, "No se han cargado datos."
        n_features = self.X.shape[1]
        n_classes = len(self.classes)
        if n_features != n_inputs or n_classes != n_outputs:
            return False, (
                f"El archivo tiene {n_features} características de entrada y {n_classes} clases, "
                f"pero la red espera {n_inputs} entradas y {n_outputs} salidas. "
                f"Configure la red con {n_features} neuronas de entrada y {n_classes} de salida."
            )
        return True, "OK"
