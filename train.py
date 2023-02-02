import csv
import sys

def read_dataset(path):
    """
    Reads the dataset from a given file path.
    Parameters:
        path(str): The file path of the dataset.

    Returns:
        None
    """
    try:
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

def estimate_price(milleage, theta0, theta1):
    return (theta0 + theta1 * milleage)

def linear_regression(numIterations, learningRate):
    """
    Performs linear regression on a given dataset.
    Parameters:
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

    # performing Geadient descent
    for j in range(numIterations):
        # Initialize tmporary values of theta0 and theta1
        tmp_theta0 = 0
        tmp_theta1 = 0
        # Loop through the dataset
        for i in range(m):
            print(prices[i], milleages[i])
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
    # print(theta0 ,theta1)
    return theta0, theta1

if __name__ == '__main__':
    milleages = []
    prices = []
    if len(sys.argv) != 2:
        print("Error: Please provide the path to the dataset file.")
        sys.exit()
    file_path = sys.argv[1]
    read_dataset(file_path)
    learningRate = 0.0001
    numIterations= 10
    print(linear_regression(numIterations, learningRate))