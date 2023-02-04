from train import estimate_price, normalize_features
import numpy as np

milleage = float(input("Enter the milleage :"))
data = np.load("min_max_theta.npz")
min_milleage= data['min']
max_milleage = data['max']
theta0 = data['theta0']
theta1 = data['theta1']
normalized_milleage = (milleage - min_milleage)/(max_milleage - min_milleage)
estimatedPrice = estimate_price(normalized_milleage, theta0, theta1)
print("The estimated price for a car of", milleage,"is",estimatedPrice)