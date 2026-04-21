import matplotlib.pyplot as plt

class Visualizador:
    @staticmethod
    def mostrar_resultados(entradas, esperados, predicciones):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        x = [f[0] for f in entradas]
        y = [f[1] for f in entradas]

        # Gráfico 1: Esperado
        axs[0].scatter(x, y, c=esperados, cmap='coolwarm')
        axs[0].set_title("Valores Originales (CSV)")

        # Gráfico 2: Predicción
        axs[1].scatter(x, y, c=predicciones, cmap='coolwarm')
        axs[1].set_title("Predicción del Perceptrón")

        # Gráfico 3: Comparativa (Aciertos/Errores)
        comparativa = ['green' if e == p else 'red' for e, p in zip(esperados, predicciones)]
        axs[2].scatter(x, y, c=comparativa)
        axs[2].set_title("Mapa de Aciertos (Verde) y Errores (Rojo)")

        plt.tight_layout()
        plt.show()