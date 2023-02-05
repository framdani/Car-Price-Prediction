from train import estimate_price, feature_scaling
import numpy as np

milleage = float(input("Enter the milleage :"))
data = np.load("min_max_theta.npz")
min_mileage= data['min_m']
max_mileage = data['max_m']
theta0 = data['theta0']
theta1 = data['theta1']
normalized_milleage = (milleage - min_mileage)/(max_mileage - min_mileage)
normalizedEstimatedPrice = estimate_price(normalized_milleage, theta0, theta1)
originalEstimatedPrice = normalizedEstimatedPrice * (data['max_p'] - data['min_p']) + data['min_p']
print("The estimated price for a car of", milleage," km is",originalEstimatedPrice)