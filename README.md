# Linear Regression for Car Price Prediction
A Linear regression model to predict the price of a car based on its mileage. The model is implemented based on gradient descent algorithm. 

## Prerequisites

- Python3
- Numpy
- Matplotlib

## Getting started

Clone the repository.
## Running the programs

### Predicting car price
To start the first program, run the following command:

```
python3 predict.py
```
If the model is already trained, the program will use the saved theta0 and theta1 values to predict the car's price for a given mileage. If the program hasn't been trained yet, the program will use the default values of theta0 and theta1 to make the prediction(both set to 0).
### Training the model
To run the second program, run the following command:
```
python3 train.py --path=path/to/dataset.csv --display=True
```
This program will read the specified dataset file (specified using `--path` argument)and perform a linear regression on the data. After the linear regression is completed, theta0 and theta1 will be saved for use in the first program. If the `--display` is set to `True`, the program will display a graph showing the data distribution and the fit line of the model.
## visualizing the results
The visualized results below gives a clear illustation of the model's performance: 
- The fit-line graph displays the relationship between the real values and the predicted values, represented by data points and the best fit-line.
- The loss over iterations graph shows the reduction of the loss function value with each iteration of the training process until it reaches the optimal solution.

<p align="center">
<img width="500" alt="fit-line" src="https://user-images.githubusercontent.com/52450718/217059637-634ede6d-ff69-4bef-a21f-fcd781fd92c0.png">
<img width="500" alt="loss-function" src="https://user-images.githubusercontent.com/52450718/217059656-fb16cc70-475d-4f67-9de7-7fc164e7b36d.png">
</p>

## Note
The learning rate and number of iterations can be changed in the train.py file.
