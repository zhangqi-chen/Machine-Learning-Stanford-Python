import numpy as np
import matplotlib.pyplot as plt


def plotData(x, y):
    """
    PLOTDATA Plots the data points x and y into a new figure
    PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit.
    """
    plt.figure()
    plt.plot(x, y, 'rx', ms=10)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.show()


def computeCost(X, y, theta):
    """
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    m = len(y)
    J = sum((np.dot(X, theta)-y)**2)/2/m
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(y)
    J_history = np.zeros([num_iters, 1])
    for i in range(num_iters):
        theta = theta-alpha/m*((np.dot(X, theta)-y)*X).sum(0, keepdims=True).T
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history
