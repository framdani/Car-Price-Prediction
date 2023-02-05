import csv
import sys
import numpy as np

def load_dataset(path='data.csv'):
    """
    Reads the dataset from a given file path.

    Parameters:
        path(str): The file path of the dataset.

    Returns:

        milleages(list): the list of milleages.
        prices(list): the list of prices.
    """
    try:
        milleages = []
        prices = []
        with open(path, "r") as file:
            reader = csv.reader(file)
            header_row = next(reader)
            try:
                milleages.append(float(header_row[0]))
                prices.append(float(header_row[1]))
            except ValueError:
                print("Header row detected, skipping...")
            for row in reader:
                milleages.append(float(row[0]))
                prices.append(float(row[1]))
    except Exception as e:
        print(f"An exception occured: {e}")
        sys.exit()
    
    print("Dataset read successfully from", path)
    return milleages, prices

def estimate_price(milleage, theta0, theta1):
    return (theta0 + theta1 * milleage)

def cost_function(X, Y, theta0, theta1):
    m = len(X)
    sqrd_error = 0
    for i in range(m):
        y_pred = theta0 + theta1 * X[i]
        sqrd_error += (y_pred - Y[i]) ** 2
    cost = 1/ (2 * m) * sqrd_error
    return cost

def gradient_descent(milleages, prices, numIterations, learningRate):
    """
    Performs linear regression on a given dataset.

    Parameters:
        mileages(list) : the list of mileages.
        prices(list) : the list of prices.
        numIterations(int) : the number of iterations the gradient descent algo will run for.
        learningRate(float) : number of steps the gradient descent algo will take.
    
    Returns:
        tuple : Estimated values of theta0 and theta1, list of cost value at each iteration.

    """
    # Number of elements in the dataset
    m = len(milleages) # or len(prices)
    # Initial values of theta0 and theta1
    theta0 = 0
    theta1 = 0
    costs = []

    # performing Gradient descent
    for epoch in range(numIterations):
        # Initialize temporary values of theta0 and theta1
        tmp_theta0 = 0
        tmp_theta1 = 0
        # Loop through the dataset
        for i in range(m):
            # Estimate the price based on the current values of theta0 and theta1
            estimatedPrice = estimate_price(milleages[i], theta0, theta1)
            # Update temporary values of theta0 and theta1
            tmp_theta0 += estimatedPrice - prices[i]
            tmp_theta1 += (estimatedPrice - prices[i]) * milleages[i]

        # Update values of theta0 and theta1 based on temporary values
        theta0 = theta0 - learningRate * (1/m) * tmp_theta0
        theta1 = theta1 - learningRate * (1/m) * tmp_theta1
        costs.append(cost_function(milleages, prices, theta0, theta1))

    return theta0, theta1, costs

def feature_scaling(data):
    Min = min(data)
    Max = max(data)
    scaling_range = Max - Min
    normalized_data = [(value - Min) / scaling_range for value in data] 
    return normalized_data, Min, Max

if __name__ == '__main__':
    # Load dataset
    if len(sys.argv) == 2:
        mileages, prices = load_dataset(sys.argv[1])
    else:
        mileages, prices = load_dataset()
    # Define hyperparameters
    learningRate  = 0.0001
    numIterations = 1000000
    # Min-Max normalization
    X_norm, min_mileage, max_mileage = feature_scaling(mileages)
    Y_norm, min_price, max_price = feature_scaling(prices)
    theta0, theta1, costs= gradient_descent(X_norm, Y_norm, numIterations, learningRate)
    np.savez("min_max_theta.npz", min=min_mileage, max = max_mileage, theta0=theta0, theta1=theta1)

    print(costs[-1]) # The min cost