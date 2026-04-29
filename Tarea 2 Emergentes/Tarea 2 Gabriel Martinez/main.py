"""
Tarea 2 — Perceptrón multicapa.
Continúa la línea de la Tarea 1 (CSV + visualización) con MLP entrenable y persistencia.
"""
from mlp import MLP
from gestor_datos import GestorDatos
from gestor_archivo_red import GestorArchivoRed
from trainer import Trainer
from visualizador import Visualizador


def imprimir_hiperparametros(mlp):
    print("\n--- Hiperparámetros ---")
    print(f"  Neuronas de entrada  (u): {mlp.n_inputs}")
    print(f"  Neuronas de salida   (v): {mlp.n_outputs}")
    print(f"  Capas ocultas        (L): {mlp.n_hidden_layers}")
    print(f"  Neuronas por capa    (b): {mlp.n_neurons_per_layer}")
    print(f"  Épocas totales       (e): {mlp.total_epochs}")
    print("-----------------------\n")


def crear_red_mlp():
    print("\n--- Nueva red MLP ---")
    u = int(input("  Neuronas de entrada:      "))
    v = int(input("  Neuronas de salida:       "))
    L = int(input("  Capas ocultas:            "))
    b = int(input("  Neuronas por capa:        "))
    return MLP(u, v, L, b)


def cargar_red_archivo():
    path = input("  Ruta del archivo: ").strip().strip('"')
    mlp = GestorArchivoRed().load(path)
    print("  Red cargada correctamente.")
    imprimir_hiperparametros(mlp)
    return mlp


def ejecutar_entrenamiento(mlp):
    path = input("  Archivo de datos de entrenamiento: ").strip().strip('"')
    epochs = int(input("  Épocas: "))
    lr_str = input("  Tasa de aprendizaje [0.05]: ").strip()
    lr = float(lr_str) if lr_str else 0.05

    gestor = GestorDatos()
    X, y = gestor.load(path)
    ok, msg = gestor.validate_dimensions(mlp.n_inputs, mlp.n_outputs)
    if not ok:
        print(f"\n  Error: {msg}\n")
        return

    trainer = Trainer(mlp, learning_rate=lr)
    print(f"\n  Entrenando por {epochs} épocas (tasa de aprendizaje={lr})...")
    epoch_errors = trainer.train(X, y, epochs)

    predictions = mlp.predict_all(X)
    Visualizador().plot_training(X, y, predictions, epoch_errors, gestor.idx_to_class)
    imprimir_hiperparametros(mlp)


def ejecutar_prueba(mlp):
    path = input("  Archivo de datos de prueba: ").strip().strip('"')

    gestor = GestorDatos()
    X, y = gestor.load(path)
    ok, msg = gestor.validate_dimensions(mlp.n_inputs, mlp.n_outputs)
    if not ok:
        print(f"\n  Error: {msg}\n")
        return

    predictions = mlp.predict_all(X)
    Visualizador().plot_testing(X, y, predictions)
    imprimir_hiperparametros(mlp)


def ejecutar_guardado(mlp):
    path = input("  Guardar en archivo: ").strip().strip('"')
    GestorArchivoRed().save(mlp, path)
    print("  Red guardada correctamente.")


def main():
    print("=" * 42)
    print("   Perceptrón multicapa (MLP)")
    print("   (extensión Tarea 1: aprendizaje)")
    print("=" * 42)
    print("  1. Crear nueva red MLP")
    print("  2. Cargar red MLP desde archivo")
    choice = input("Elija una opción: ").strip()

    if choice == '1':
        mlp = crear_red_mlp()
    elif choice == '2':
        mlp = cargar_red_archivo()
    else:
        print("Opción inválida.")
        return

    while True:
        print("\n--- Menú ---")
        print("  1. Entrenar  (retropropagación)")
        print("  2. Evaluar   (alimentación hacia adelante)")
        print("  3. Guardar red")
        print("  4. Salir")
        choice = input("Elija una opción: ").strip()

        if choice == '1':
            ejecutar_entrenamiento(mlp)
        elif choice == '2':
            ejecutar_prueba(mlp)
        elif choice == '3':
            ejecutar_guardado(mlp)
        elif choice == '4':
            print("Hasta luego.")
            break
        else:
            print("Opción inválida.")


if __name__ == '__main__':
    main()
