import csv
import matplotlib.pyplot as plt

# --- FUNCIONES NÚCLEO (DESDE CERO) ---

def cargar_datos(nombre_archivo):
    entradas = []
    esperados = []
    try:
        with open(nombre_archivo, mode='r', encoding='utf-8-sig') as archivo:
            lector_csv = csv.reader(archivo)
            
            # Saltamos la primera fila (cabecera: x1, x2, y)
            next(lector_csv, None) 
            
            for fila in lector_csv:
                if not fila: continue
                # Ahora sí convertimos a float
                valores = [float(x) for x in fila]
                entradas.append(valores[:-1])
                esperados.append(valores[-1])
        return entradas, esperados
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None, None

def calcular_suma(entradas_fila, pesos, sesgo):
    """Implementa la función suma solicitada"""
    total = sesgo 
    for i in range(len(entradas_fila)):
        total += entradas_fila[i] * pesos[i]
    return total

def activacion_escalon(z):
    """Función de activación tipo escalón"""
    return 1 if z >= 0 else 0

def activacion_signo(z):
    """Función de activación tipo signo"""
    return 1 if z >= 0 else -1

# --- INTERFAZ DE USUARIO ---

def pedir_configuracion(n_entradas):
    """Lee del usuario los pesos y la función de activación """
    print("\n--- Configuración de Pesos ---")
    pesos = []
    for i in range(n_entradas):
        p = float(input(f"Ingrese peso para entrada x{i+1}: ")) 
        pesos.append(p)
    
    sesgo = float(input("Ingrese peso para el sesgo (bias): ")) 
    
    print("\nFunciones de activación:")
    print("1. Escalón (0 o 1)")
    print("2. Signo (-1 o 1)")
    opcion = input("Seleccione función (1 o 2): ") 
    
    return pesos, sesgo, opcion

def generar_graficos(entradas, esperados, predichos):
    # Si n > 3, solo graficamos las primeras dos dimensiones
    x_coords = [fila[0] for fila in entradas]
    y_coords = [fila[1] for fila in entradas]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Análisis del Perceptrón')

    # Gráfico 1: Valor Esperado 
    ax1.scatter(x_coords, y_coords, c=esperados, cmap='winter')
    ax1.set_title('Valor Esperado (CSV)')

    # Gráfico 2: Valor Predicho 
    ax2.scatter(x_coords, y_coords, c=predichos, cmap='winter')
    ax2.set_title('Predicción del Perceptrón')

    # Gráfico 3: Comparativa (Aciertos/Errores) 
    # Verde si coinciden, Rojo si no coinciden 
    colores = ['green' if e == p else 'red' for e, p in zip(esperados, predichos)]
    ax3.scatter(x_coords, y_coords, c=colores)
    ax3.set_title('Mapa de Aciertos (V) y Errores (R)')

    plt.tight_layout()
    plt.show()

# --- 4. MENÚ Y EJECUCIÓN ---
def main():
    print("Tarea 1: Perceptrón - Computación Emergente")
    ruta = input("Ingrese el nombre del archivo CSV: ")
    entradas, esperados = cargar_datos(ruta)
    
    if entradas is None: return
    n_entradas = len(entradas[0])

    while True:
        # Pedir pesos y función al usuario 
        pesos = []
        for i in range(n_entradas):
            p = float(input(f"Peso para entrada x{i+1}: "))
            pesos.append(p)
        
        sesgo = float(input("Peso para el sesgo (bias): "))
        
        print("\n1. Escalón\n2. Signo")
        opc = input("Seleccione función de activación: ")

        # Proceso de predicción
        predicciones = []
        aciertos = 0

        # Usamos enumerate para tener el índice 'i' y poder acceder a esperados[i]
        for i, fila in enumerate(entradas):
            z = calcular_suma(fila, pesos, sesgo)
            
            # 1. Obtener la predicción del perceptrón
            y_pred = activacion_escalon(z) if opc == "1" else activacion_signo(z)
            predicciones.append(y_pred)
            
            # 2. Redondear el valor del CSV para manejar el ruido (0.2 -> 0)
            # Esto es vital para que te marque acierto en el archivo separable
            valor_esperado_real = int(round(esperados[i])) 
            
            # 3. Comparar
            if y_pred == valor_esperado_real:
                aciertos += 1
                status = "✓"
            else:
                status = "✗"
            
            # Imprimir detalle (opcional, puedes comentarlo si son muchas filas)
            if i < 20: # Solo mostramos las primeras 20 para no saturar la consola
                print(f"Entrada: {fila}, Suma: {z:.2f}, Predicción: {y_pred}, Esperado Real: {valor_esperado_real} {status}")

        # Resultados finales
        precision = (aciertos / len(esperados)) * 100
        print(f"\n--- RESULTADOS ---")
        print(f"Aciertos: {aciertos} de {len(esperados)}")
        print(f"Precisión total: {precision:.2f}%")
        
        # Generar gráficos (solo una vez)
        print("Generando gráficos... (Cierra la ventana para continuar)")
        # Creamos una lista de los esperados PERO redondeados para que coincidan con la lógica del 100%
        esperados_limpios = [int(round(e)) for e in esperados]

        # Llamamos al gráfico usando los datos redondeados
        generar_graficos(entradas, esperados_limpios, predicciones)
        

        #Pesos correctos: 1.5 , -1.5, -0.37
        # Permitir probar con otros pesos
        if input("\n¿Desea probar otros pesos? (s/n): ").lower() != 's':
            break

if __name__ == "__main__":
    main()