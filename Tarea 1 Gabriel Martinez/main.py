from perceptron import Perceptron
from gestor_datos import GestorDatos
from visualizador import Visualizador

def ejecutar_sistema():
    print("--- Sistema de Perceptrón Simple (POO) ---")
    ruta = input("Nombre del archivo CSV: ")
    entradas, esperados = GestorDatos.cargar_csv(ruta)

    if entradas:
        # Instanciamos nuestra neurona
        modelo = Perceptron(len(entradas[0]))

        while True:
            # Configuración de pesos
            for i in range(len(modelo.pesos)):
                modelo.pesos[i] = float(input(f"Peso x{i+1}: "))
            modelo.sesgo = float(input("Sesgo (bias): "))
            
            print("\n1. Escalón\n2. Signo")
            opc = input("Selección: ")

            predicciones_finales = []
            aciertos = 0

            for i, fila in enumerate(entradas):
                y_pred, z = modelo.predecir(fila, opc)
                predicciones_finales.append(y_pred)
                
                if y_pred == esperados[i]:
                    aciertos += 1
                    status = "✓"
                else:
                    status = "✗"
                
                if i < 10: # Mostrar solo algunos para no saturar
                    print(f"Z: {z:.2f} | Pred: {y_pred} | Real: {esperados[i]} {status}")

            precision = (aciertos / len(esperados)) * 100
            print(f"\n>>> PRECISIÓN FINAL: {precision:.2f}%")
                        #Pesos correctos: 1.5 , -1.5, -0.37

            Visualizador.mostrar_resultados(entradas, esperados, predicciones_finales)

            if input("\n¿Probar otros pesos? (s/n): ").lower() != 's':
                break

if __name__ == "__main__":
    ejecutar_sistema()

