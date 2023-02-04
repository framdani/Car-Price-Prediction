# Linear Regression for Car Price Prediction
A Linear regression model to predict the price of a car based on its mileage. The model is implemented based on gradient descent algorithm. 

## Prerequisites
- Python3
- Numpy libarary
## Getting started
Clone the repository.

## Running the programs
### Predicting car price
To start the first program, run the following command:

```
python3 predict.py
```
If the model is already trained, the program will use theta0 and theta1 values to predict th price of a car for a given mileage. If the program hasn't been trained yet, the program will use the default values of theta0 and theta1 to make the prediction(both set to 0).
### Training the model
To run the second program, run the following command:
```
python3 train.py [path/to/dataset.csv]
```
This program will read the specified dataset file and perform a linear regression on the data. After the linear regression is completed, theta0 and theta1 will be saved for use in the first program.
## Note
The learning rate, number of iterations can be changed in the train.py file.
