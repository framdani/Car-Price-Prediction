from train import estimate_price, feature_scaling
import numpy as np

milleage = float(input("Enter the milleage :"))
data = np.load("min_max_theta.npz")
min_mileage= data['min']
max_mileage = data['max']
theta0 = data['theta0']
theta1 = data['theta1']
normalized_milleage = (milleage - min_mileage)/(max_mileage - min_mileage)
normalizedEstimatedPrice = estimate_price(normalized_milleage, theta0, theta1)
originalEstimatedPrice = normalizedEstimatedPrice * (max_mileage - min_mileage) + min_mileage
print("The estimated price for a car of", milleage,"is",originalEstimatedPrice)