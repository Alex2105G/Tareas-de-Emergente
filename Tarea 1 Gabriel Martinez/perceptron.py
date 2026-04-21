class Perceptron:
    def __init__(self, n_entradas):
        # Inicializamos con valores que el usuario cambiará luego
        self.pesos = [0.0] * n_entradas
        self.sesgo = 0.0

    def calcular_suma(self, entrada):
        suma = self.sesgo
        for i in range(len(entrada)):
            suma += entrada[i] * self.pesos[i]
        return suma

    def activar_escalon(self, z):
        return 1 if z >= 0 else 0

    def activar_signo(self, z):
        return 1 if z >= 0 else -1

    def predecir(self, entrada, tipo_activacion):
        z = self.calcular_suma(entrada)
        if tipo_activacion == "1":
            return self.activar_escalon(z), z
        else:
            return self.activar_signo(z), z