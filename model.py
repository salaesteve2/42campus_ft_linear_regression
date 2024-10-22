import argparse
import sys
import os
import time
import json

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def load_parameters():
	try:
		with open('model.json', 'r') as file:
			parameters = json.load(file)
			theta0 = parameters.get('theta0', 0)
			theta1 = parameters.get('theta1', 0)
	except FileNotFoundError:
		theta0 = 0
		theta1 = 0
	return theta0, theta1


def main():
	theta0, theta1 = load_parameters()

	mileage = float(input('Introduce el kilometraje del coche: '))
	price = estimate_price(mileage, theta0, theta1)

	print(f"El precio estimado del coche es: {price:.2f} euros.")




if __name__ == "__main__":
    main()