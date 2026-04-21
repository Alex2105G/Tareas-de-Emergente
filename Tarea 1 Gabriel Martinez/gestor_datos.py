import csv

class GestorDatos:
    @staticmethod
    def cargar_csv(ruta):
        entradas = []
        esperados = []
        try:
            with open(ruta, 'r', encoding='utf-8-sig') as f:
                lector = csv.reader(f)
                for fila in lector:
                    # Convertimos x1, x2 a float y la salida y a int redondeado
                    entradas.append([float(fila[0]), float(fila[1])])
                    esperados.append(int(round(float(fila[2]))))
            return entradas, esperados
        except FileNotFoundError:
            print("Error: Archivo no encontrado.")
            return None, None