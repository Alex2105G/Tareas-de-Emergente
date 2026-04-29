import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class Visualizador:

    def _scatter(self, ax, X, colors, title, c_values=None, cmap='tab10'):
        if X.shape[1] >= 3:
            sc = ax.scatter(
                X[:, 0], X[:, 1], X[:, 2],
                c=colors if c_values is None else c_values,
                cmap=cmap if c_values is not None else None,
            )
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')
        else:
            sc = ax.scatter(
                X[:, 0], X[:, 1],
                c=colors if c_values is None else c_values,
                cmap=cmap if c_values is not None else None,
            )
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
        ax.set_title(title)
        return sc

    def _make_fig(self, X):
        fig = plt.figure(figsize=(16, 5))
        is_3d = X.shape[1] >= 3
        proj = '3d' if is_3d else None
        ax1 = fig.add_subplot(1, 3, 1, projection=proj)
        ax2 = fig.add_subplot(1, 3, 2, projection=proj)
        ax3 = fig.add_subplot(1, 3, 3)
        return fig, ax1, ax2, ax3

    def plot_training(self, X, y, predictions, epoch_errors, idx_to_class=None):
        predictions = np.array(predictions)
        correct = predictions == np.array(y)
        colors = ['green' if c else 'red' for c in correct]
        accuracy = np.mean(correct)

        fig, ax1, ax2, ax3 = self._make_fig(X)

        sc = self._scatter(ax1, X, None, 'Datos', c_values=y, cmap='tab10')
        plt.colorbar(sc, ax=ax1)

        self._scatter(ax2, X, colors, f'Predicciones\nExactitud: {accuracy:.2%}')

        ax3.plot(range(1, len(epoch_errors) + 1), epoch_errors, color='blue')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Error')
        ax3.set_title('Error de Entrenamiento por Época')

        plt.tight_layout()
        plt.show()

    def plot_testing(self, X, y, predictions):
        y = np.array(y)
        predictions = np.array(predictions)
        correct = predictions == y
        colors = ['green' if c else 'red' for c in correct]
        accuracy = np.mean(correct)

        n_classes = int(max(np.max(y), np.max(predictions))) + 1
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y, predictions):
            cm[t][p] += 1

        fig, ax1, ax2, ax3 = self._make_fig(X)

        sc = self._scatter(ax1, X, None, 'Datos', c_values=y, cmap='tab10')
        plt.colorbar(sc, ax=ax1)

        self._scatter(ax2, X, colors, f'Predicciones\nExactitud: {accuracy:.2%}')

        mat = ax3.matshow(cm, cmap='Blues')
        plt.colorbar(mat, ax=ax3)
        ax3.set_xlabel('Predicho')
        ax3.set_ylabel('Real')
        ax3.set_title('Matriz de Confusión')
        ax3.xaxis.set_label_position('bottom')
        ax3.xaxis.tick_bottom()
        for i in range(n_classes):
            for j in range(n_classes):
                ax3.text(
                    j, i, str(cm[i, j]), ha='center', va='center',
                    color='black' if cm[i, j] < cm.max() / 2 else 'white',
                )

        plt.tight_layout()
        plt.show()
