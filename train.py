import csv
import sys
import numpy as np

def read_dataset(path):
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
    except FileNotFoundError:
        print("Error: The file", path, "was not found.")
        sys.exit()
    except:
        print("Error: An unknown error occured while reading the file.")
        sys.exit()
    
    print("Dataset read successfully from", path)
    return milleages, prices

def estimate_price(milleage, theta0, theta1):
    return (theta0 + theta1 * milleage)

def cost_function(X, y, theta0, theta1):
    m = len(X)
    sqrd_error = 0
    for i in range(m):
        y_pred = theta0 + theta1 * X[i]
        sqrd_error += (y_pred - y[i]) ** 2
    cost = 1/ (2 * m) * sqrd_error
    return cost

def gradient_descent(milleages, prices, numIterations, learningRate):
    """
    Performs linear regression on a given dataset.
    Parameters:
        milleages(list) : the list of milleages.
        prices(list) : the list of prices.
        numIterations(int) : the number of iterations the gradient descent algo will run for.
        learningRate(float) : number of steps the gradient descent algo will take.
    
    Returns:
        tuple : the estimated values of theta0 and theta1

    """
    # Number of elements in the dataset
    m = len(milleages) # or len(prices)
    # Initial values of theta0 and theta1
    theta0 = 0
    theta1 = 0
    #X_norm = normalize_features(milleages)

    # performing Gradient descent
    for j in range(numIterations):
        # Initialize temporary values of theta0 and theta1
        tmp_theta0 = 0
        tmp_theta1 = 0

        costs = []
        # Loop through the dataset
        for i in range(m):
            # print(prices[i], milleages[i])
            # Estimate the price based on the current values of theta0 and theta1
            estimatedPrice = estimate_price(milleages[i], theta0, theta1)
            print("estimadPrice",estimatedPrice)
            # Update temporary values of theta0 and theta1
            tmp_theta0 += estimatedPrice - prices[i]
            tmp_theta1 += (estimatedPrice - prices[i]) * milleages[i]
            print('tmp theta0',tmp_theta0, 'tmp theta1 ',tmp_theta1)

        # Update values of theta0 and theta1 based on temporary values
        theta0 = theta0 - learningRate * (1/m) * tmp_theta0
        theta1 = theta1 - learningRate * (1/m) * tmp_theta1
        costs.append(cost_function(milleages,prices, theta0, theta1))
    # print(theta0 ,theta1)
    return theta0, theta1, costs

def normalize_features(milleages):
    # mean_milleages = sum(milleages)/len(milleages)
    # std_millages = (sum((x - mean_milleages) ** 2 for x in milleages)/ len(milleages))**0.5
    # milleages_normalized = [(x - mean_milleages)/std_millages for x in milleages]
    # return milleages_normalized, mean_milleages, std_millages
    min_milleage = min(milleages)
    max_milleage = max(milleages)
    range = max_milleage - min_milleage
    milleages_normalized = [(x - min_milleage) / range for x in milleages] 
    return milleages_normalized, min_milleage, max_milleage

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the dataset file.")
        sys.exit()
    file_path = sys.argv[1]
    milleages, prices = read_dataset(file_path)
    learningRate  = 0.01
    numIterations = 1000
    X_norm, min_milleage, max_milleage = normalize_features(milleages)
    theta0, theta1, costs= gradient_descent(X_norm, prices, numIterations, learningRate)
    np.savez("min_max_theta.npz", min=min_milleage, max = max_milleage, theta0=theta0, theta1=theta1)
    print(costs)


# 