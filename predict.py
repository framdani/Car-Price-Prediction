from train import estimate_price
import numpy as np


try:
    # Get the input mileage form the user
    while True:
        mileage = input("Enter the mileage :")
        try:
            mileage = float(mileage)
            if mileage >= 0:
                break
            print("Invalid input. Mileage must be positive value.")
        except:
            print("Invalid input. Mileage must be a number.")
    
    # Load the saved min-max values and theta parameters
    data = np.load("min_max_theta.npz")
    min_mileage= data['min_m']
    max_mileage = data['max_m']
    theta0 = data['theta0']
    theta1 = data['theta1']
    
    # Normalize the mileage
    normalized_milleage = (mileage - min_mileage)/(max_mileage - min_mileage)

    # Get the normalized estimated price
    normalizedEstimatedPrice = estimate_price(normalized_milleage, theta0, theta1)

    #Denormalize the estimated price to get the original one
    originalEstimatedPrice = normalizedEstimatedPrice * (data['max_p'] - data['min_p']) + data['min_p']
    if originalEstimatedPrice < 0:
        print("Warning : The predicted price is negative. This is likely due to extrapolation beyond the range of the training data.")
    else:
        print("The estimated price for a car of", mileage," km is",originalEstimatedPrice)  
except FileNotFoundError:
    theta0 = 0
    theta1 = 0
    print("The estimated price for a car of", mileage," km is",estimate_price(mileage, theta0, theta1))


