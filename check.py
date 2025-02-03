import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib  # Para guardar y cargar el modelo

MODEL_FILE = "linear_model.pkl"

def train_and_predict():
    """Entrena un modelo de regresión lineal y realiza predicciones"""
    
    # Cargar datos
    data = pd.read_csv('data.csv')
    X = data['km'].values.reshape(-1, 1)  # Kilometraje (feature)
    y = data['price'].values  # Precio (target)

    # Visualizar los datos
    plt.scatter(X, y, color='blue', label='Datos reales')
    plt.xlabel('Kilometraje')
    plt.ylabel('Precio')
    plt.title('Relación entre Kilometraje y Precio')
    plt.legend()
    plt.show()

    # Verificar si el modelo ya ha sido entrenado
    if os.path.exists(MODEL_FILE):
        print("Cargando modelo previamente entrenado...")
        model = joblib.load(MODEL_FILE)
    else:
        print("Entrenando un nuevo modelo de regresión lineal...")
        model = LinearRegression()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)  # Guardar el modelo entrenado

    # Predicciones en el conjunto de datos
    y_pred = model.predict(X)

    # Evaluación del modelo
    precision = mean_absolute_error(y, y_pred)
    percentage = precision / (sum(y) / len(y)) * 100
    print(f"El error promedio es: {precision:.2f}")
    print(f"El modelo tiene un {100 - percentage:.2f}% de precisión.")

    # Graficar predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.plot(X, y_pred, color='red', linestyle='-', label='Regresión Lineal')
    plt.xlabel("Kilometraje")
    plt.ylabel("Precio")
    plt.title("Comparación entre Datos Reales y Predicciones")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

def predict_price(mileage_input, model):
    """Realiza una predicción del precio basado en el kilometraje ingresado"""
    mileage_array = [[mileage_input]]  # Convertir a array bidimensional
    return model.predict(mileage_array)[0]

if __name__ == "__main__":
    model = train_and_predict()

    while True:
        try:
            mileage_input = input("Ingrese el kilometraje (o 'salir' para terminar): ").strip().lower()
            if mileage_input == 'salir':
                print("Saliendo del programa.")
                break
            
            mileage_input = float(mileage_input)
            if mileage_input < 0:
                print("Error: kilometraje negativo. Intente nuevamente.")
                continue
            
            estimated_price = predict_price(mileage_input, model)
            print(f"El precio estimado para un kilometraje de {mileage_input} es: {estimated_price:.2f}")
            
        except ValueError:
            print("Error: no es un número válido. Intente nuevamente.")
        except KeyboardInterrupt:
            print("\nSaliendo del programa. ¡Hasta luego!")
            break
