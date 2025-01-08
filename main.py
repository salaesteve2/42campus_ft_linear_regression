import os
import pandas as pd
import matplotlib.pyplot as plt
from training import train_model, load_parameters, estimatePrice, mean_absolute_error

def main():
    # Cargar datos
    data = pd.read_csv('data.csv')
    mileage = data['km'].tolist()  # Kilometraje
    price = data['price'].tolist()  # Precio

    # Visualizar los datos para verificar la relación entre kilometraje y precio
    plt.scatter(mileage, price, color='blue', label='Datos reales')
    plt.xlabel('Kilometraje')
    plt.ylabel('Precio')
    plt.title('Relación entre Kilometraje y Precio')
    plt.legend()
    plt.show()

    # Archivo donde se guardan los parámetros
    parameters_file = 'parameters.json'

    # Comprobamos si los parámetros están guardados
    if not os.path.exists(parameters_file):
        print("No se encontraron parámetros guardados.")
        train_choice = input("¿Quieres entrenar el modelo? (sí/no): ").strip().lower()
    else:
        train_choice = 'no'  # Si ya existen parámetros, no preguntar

    # Si no existen parámetros guardados, entrenamos el modelo
    if train_choice == 'sí' or train_choice == 'si':
        print("Entrenando el modelo...")
        # Entrenar el modelo y guardar los parámetros normalizados
        theta0, theta1, mean_x, std_x, mean_y, std_y = train_model(mileage, price)
        print(f"Modelo entrenado: theta0 = {theta0}, theta1 = {theta1}")
    else:
        # Cargar parámetros guardados
        theta0, theta1, mean_x, std_x, mean_y, std_y = load_parameters(parameters_file)
        
        if theta0 == 0 and theta1 == 0:
            print("No se encontraron parámetros guardados. Iniciando en 0.")
        else:
            print("Usando parámetros cargados.")

    y_true = price
    y_pred = [estimatePrice((mileage[i] - mean_x) / std_x, theta0, theta1) * std_y + mean_y for i in range(len(mileage))]
    print(y_pred)
    precision = mean_absolute_error(y_true, y_pred)
    percentage = precision / (sum(price) / len(price)) * 100
    print(f"El error promedio es: {precision:.2f}")
    print(f"El modelo tiene un {100 - percentage:.2f}% de precisión.")

    # Crear una figura y un conjunto de ejes
    plt.figure(figsize=(10, 6))

    # Graficar los datos reales
    plt.scatter(mileage, price, color='blue', label='Datos Reales')

    # Graficar las predicciones
    plt.plot(mileage, y_pred, color='red', label='Predicciones del Modelo', linestyle='-')

    # Añadir títulos y etiquetas
    plt.title("Comparación entre Datos Reales y Predicciones", fontsize=14)
    plt.xlabel("Kilometraje", fontsize=12)
    plt.ylabel("Precio", fontsize=12)

    # Mostrar la leyenda
    plt.legend()

    # Mostrar la gráfica
    plt.grid(True)
    plt.show()

    # Bucle para solicitar predicciones
    while True:
        try:
            mileage_input = input("Ingrese el kilometraje (o escribe 'salir' para terminar): ").strip().lower()
            if mileage_input == 'salir':
                print("Saliendo del programa.")
                break
            
            # Verificamos si el input es un número válido
            mileage_input = float(mileage_input)
            if mileage_input < 0:
                print("Error: kilometraje negativo. Intente nuevamente.")
                continue
            
            # Predecimos el precio normalizado
            mileage_normalized = (mileage_input - mean_x) / std_x

    # Predicción en datos normalizados
            y_pred_normalized = estimatePrice(mileage_normalized, theta0, theta1)

            # Revertir normalización de la predicción
            y_pred_original = y_pred_normalized * std_y + mean_y
            
            
            print(f"El precio estimado para un kilometraje de {mileage_input} es: {y_pred_original:.2f}")
            
        except ValueError:
            print("Error: no es un número válido. Intente nuevamente.")
        except KeyboardInterrupt:
            print("\nSaliendo del programa. ¡Hasta luego!")
            break

if __name__ == "__main__":
    main()
