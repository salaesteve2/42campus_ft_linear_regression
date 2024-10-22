import numpy as np
import pandas as pd
import json

def compute_cost(mileage, price, theta0, theta1):
    m = len(mileage)
    predictions = theta0 + theta1 * mileage
    cost = (1 / (2 * m)) * np.sum((predictions - price) ** 2)
    return cost

def gradient_descent(mileage, price, theta0, theta1, learning_rate, iterations):
    m = len(mileage)
    for i in range(iterations):
        predictions = theta0 + theta1 * mileage
        d_theta0 = (1 / m) * np.sum(predictions - price)
        d_theta1 = (1 / m) * np.sum((predictions - price) * mileage)
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1
        
        # Calcular y mostrar el costo cada 100 iteraciones
        if i % 100 == 0:
            cost = compute_cost(mileage, price, theta0, theta1)
            print(f"Iteración {i}: Costo = {cost:.4f}, θ0 = {theta0:.4f}, θ1 = {theta1:.4f}")
    
    return theta0, theta1

def normalize(data):
    """ Normaliza los datos para que estén en el rango [0, 1] """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def train_model(mileage, price, learning_rate=0.01, iterations=1000):
    theta0, theta1 = 0, 0  # Inicialización
    theta0, theta1 = gradient_descent(mileage, price, theta0, theta1, learning_rate, iterations)
    return theta0, theta1

def save_parameters(theta0, theta1):
    parameters = {
        'theta0': theta0,
        'theta1': theta1
    }
    with open('model.json', 'w') as file:
        json.dump(parameters, file)

def main():
    # Cargar los datos desde el archivo CSV
    data = pd.read_csv('data.csv')  # Asegúrate de que la ruta sea correcta
    mileage = data['km'].values
    price = data['price'].values

    # Normalizar los datos de kilometraje
    mileage_normalized = normalize(mileage)

    learning_rate = 0.01
    iterations = 10000

    theta0, theta1 = train_model(mileage_normalized, price, learning_rate, iterations)
    save_parameters(theta0, theta1)
    
    print(f"Entrenamiento completo: θ0 = {theta0}, θ1 = {theta1}")

if __name__ == "__main__":
    main()



