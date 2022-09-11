import numpy as np 
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class Regression:
    
    def __init__(self, predictors, criterions):
        # predictors and criterions are matrices with data
        if isinstance(predictors, list):
            self.x = np.array(predictors)
        elif isinstance(predictors, np.ndarray):
            self.x = predictors
        else:
            raise TypeError('Neither a list nor a numpy.array')

        if isinstance(criterions, list):
            self.y = np.array(criterions)
        elif isinstance(criterions, np.ndarray):
            self.y = criterions
        else:
            raise TypeError('Neither a list nor a numpy.array')

        # creating a new matrix with a column of ones on the right
        self.X = np.hstack((x, np.ones((x.shape[0], 1))))
        # random initial values for constants (they will be changed during the
        # learning process)
        self.constants = np.random.randn(2, 1)
        self._showable = False

    # calculating everything for the given dataset and parameters
    # refer to methods below for maths
    def calculate(self, learning_rate, iterations):
        self.final_theta, self.cost_history = gradient_descent(self.X, self.y, self.constants, learning_rate, iterations)
        self.predictions = model(self.X, self.final_theta)
        self.R = coef_determination(self.y, self.predictions)
        self._showable = True

    # showing data points and calculated fit with matplotlib 
    def show(self):
        if self._showable:
            fig, ax = plt.subplots()
            ax.scatter(self.x, self.y)
            plt.plot(self.x, self.predictions)
            plt.show()
        else:
            raise RuntimeError('You need to calculate first')

    # showing the learning curve
    def show_learning(self, iterations):
        if self._showable:
            plt.clf()
            plt.plot(range(iterations), self.cost_history)
            print(self.R)
            plt.show()
        else:
            raise RuntimeError('You need to calculate first')

#######################################################################
#              MATHS
#######################################################################

# product between matrix with variables (data) and constants (that we wanna find)
# this is basically a polynomial function
# the resulting matrix contains all of the images
def model(predictors, constants):
    return predictors.dot(constants)

# using mean squared error we calculate how good our model is
def cost_function(predictors, criterions, constants):
    m = len(criterions)
    return 1 /(2 * m) * np.sum((model(predictors, constants) - criterions)**2)

# calculation of the derivative of the cost function with respect of the
# constants we are trying to find
def gradient(predictors, criterions, constants):
    m = len(criterions)
    return 1/m * predictors.T.dot(model(predictors, constants) - criterions)

# algorithm to reach the minimum of the cost function (the best constants)
# also stores all of the cost values to keep track of the learning process
def gradient_descent(predictors, criterions, constants, learning_rate, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        constants -= learning_rate * gradient(predictors, criterions, constants)
        cost_history[i] = cost_function(predictors, criterions, constants)
    return constants, cost_history

# formula for the coefficient of determination (the closer to one, the better)
def coef_determination(criterions, predictions):
    num = ((criterions - predictions)**2).sum()
    den = ((criterions - criterions.mean())**2).sum()
    return 1 - num/den

# generating random data
x, y = make_regression(n_features = 1, n_samples = 50, noise = 10)
y = y.reshape(y.shape[0], 1)
