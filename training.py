import os
import json

def estimatePrice(mileage, theta_0, theta_1):
    return theta_0 + theta_1 * mileage

def compute_cost(X, y, theta_0, theta_1):
    m = len(y)  # Número de datos
    total_cost = 0.0
    for i in range(m):
        prediction = theta_0 + theta_1 * X[i]
        error = prediction - y[i]
        total_cost += error ** 2
    return total_cost / (2 * m)

def mean_absolute_error(y_true, y_pred):
    """Calcula el Error Absoluto Medio (MAE)"""
    total_error = 0.0
    n = len(y_true)
    for i in range(n):
        total_error += abs(y_true[i] - y_pred[i])
    return total_error / n

def gradient_descent(X, y, theta_0, theta_1, learning_rate, iterations):
    # theta0 = learning_rate * (1 / m) * sum(estimatePrice(mileage[i]) - price[i])
    # theta1 = learning_rate * (1 / m) * sum((estimatePrice(mileage[i]) - price[i]) * mileage[i])
    # Donde m es el número de datos

    # tmp00 = 0.0001 * (1 / len(y)) * (for i in range[iterations]{sum += estimatePrice(x[i], theta0, theta1) sum -= y[i] })   
    # tmp01 = 0.0001 * (1 / len(y)) * (for i in range[iterations]{sum += (estimatePrice(x[i], theta0, theta1) sum -= y[i]) * x[i] })

    m = len(y)  # Número de datos
    for iteration in range(iterations):
        sum_error_theta_0 = 0.0
        sum_error_theta_1 = 0.0
        
        # Calculamos los gradientes
        for i in range(m):
            prediction = theta_0 + theta_1 * X[i]
            error = prediction - y[i]
            sum_error_theta_0 += error
            sum_error_theta_1 += error * X[i]
        
        # Actualizamos los parámetros
        theta_0 -= learning_rate * (1 / m) * sum_error_theta_0
        theta_1 -= learning_rate * (1 / m) * sum_error_theta_1
        
        # Imprimimos el costo cada 100 iteraciones
        if iteration % 100 == 0:
            cost = compute_cost(X, y, theta_0, theta_1)
            print(f"Iteración {iteration}: Costo = {cost}, theta_0 = {theta_0}, theta_1 = {theta_1}")
    
    return theta_0, theta_1

def save_parameters(theta0, theta1, mean_x, std_x, mean_y, std_y, filename='parameters.json'):
    with open(filename, 'w') as f:
        json.dump({
            'theta0': theta0,
            'theta1': theta1,
            'mean_x': mean_x,
            'std_x': std_x,
            'mean_y': mean_y,
            'std_y': std_y
        }, f)

def load_parameters(filename='parameters.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            params = json.load(f)
        return (params['theta0'], params['theta1'], 
                params['mean_x'], params['std_x'], 
                params['mean_y'], params['std_y'])
    else:
        return 0, 0  # Valores predeterminados

def train_model(mileage, price):
    theta_0 = 0
    theta_1 = 0

    # Parámetros de gradiente descendente
    learning_rate = 0.001
    iterations = 10000
    print(mileage)
    print(price)

    mean_x = sum(mileage) / len(mileage)
    std_x = (sum((xi - mean_x) ** 2 for xi in mileage) / len(mileage)) ** 0.5
    mileage_normalized = [(xi - mean_x) / std_x for xi in mileage]

    mean_y = sum(price) / len(price)
    std_y = (sum((yi - mean_y) ** 2 for yi in price) / len(price)) ** 0.5
    price_normalized = [(yi - mean_y) / std_y for yi in price]

    theta_0, theta_1 = gradient_descent(mileage_normalized, price_normalized, theta_0, theta_1, learning_rate, iterations)

    print(f"Valores de theta_0 y theta_1 después de {iterations} iteraciones:")
    print(f"theta_0 = {theta_0}, theta_1 = {theta_1}")
    
    # Guardamos los parámetros y los valores de normalización
    save_parameters(theta_0, theta_1, mean_x, std_x, mean_y, std_y)
    
    return theta_0, theta_1, mean_x, std_x, mean_y, std_y
