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
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta-alpha/m*((np.dot(X, theta)-y)*X).sum(0, keepdims=True).T
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


def featureNormalize(X):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    """
    COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
    J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    m = len(y)
    diff = np.dot(X, theta)-y
    J = np.dot(diff.T, diff)/2/m  # J=(X*theta-y)'*(X*theta-y)/2m
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        deriv = np.dot(X.T, np.dot(X, theta)-y)  # dJ/dtheta=X'*(X*theta-y)
        theta = theta-alpha/m*deriv
        J_history[i] = computeCost(X, y, theta)
    return theta, J_history


def normalEqn(X, y):
    """
    NORMALEQN Computes the closed-form solution to linear regression
    NORMALEQN(X,y) computes the closed-form solution to linear
    regression using the normal equations.
    """
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # inv(X'*X)*X'*y
    return theta
