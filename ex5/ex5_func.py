import numpy as np
import matplotlib.pyplot as plt


def linearRegCostFunction(X, y, theta, lbd):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    regression with multiple variables
    [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    m = len(y)
    # J=(1/2m)*sum(X*theta-y)+(lambda/2m)*sum(theta^2)
    h = X.dot(theta)
    J = ((h-y)**2).sum()/2/m
    reg = (theta[1:]**2).sum()/2/m
    J = J+lbd*reg
    # dJ/dtheta=(1/m)*X'*(X*theta-y)+(lambda/m)*theta
    grad = X.T.dot(h-y)/m
    grad[1:] += lbd/m*theta[1:]

    return J, grad


def trainLinearReg(X, y, theta, alpha, num_iters, lbd):
    """
    Using gradient descent to train data
    """
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = linearRegCostFunction(X, y, theta, lbd)
        theta = theta-alpha*grad
    return theta, J_history


def learningCurve(X, y, Xval, yval, alpha, num_iters, lbd):
    """
    LEARNINGCURVE Generates the train and cross validation set errors needed
    to plot a learning curve
    [error_train, error_val] = ...
        LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
        cross validation set errors for a learning curve. In particular,
        it returns two vectors of the same length - error_train and
        error_val. Then, error_train(i) contains the training error for
        i examples (and similarly for error_val(i)).
    """
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        X_train = X[:i+1]
        y_train = y[:i+1]
        theta = np.zeros((X.shape[1], 1))
        theta, _ = trainLinearReg(X_train, y_train, theta, alpha, num_iters, lbd)
        error_train[i], _ = linearRegCostFunction(X_train, y_train, theta, lbd)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, lbd)

    return error_train, error_val


def polyFeatures(X, p):
    """
    POLYFEATURES Maps X (1D vector) into the p-th power
    [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    """
    X_poly = np.zeros((X.size, p))
    for i in range(p):
        X_poly[:, [i]] = X**(i+1)
    return X_poly


def featureNormalize(X):
    """
    FEATURENORMALIZE Normalizes the features in X
    FEATURENORMALIZE(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """
    PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
    """
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array([np.arange(min_x-15, max_x+25.01, 0.05)]).T

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly-mu)/sigma

    # Add ones
    X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

    # Plot
    plt.plot(x, X_poly.dot(theta), '--')


def validationCurve(X, y, Xval, yval, alpha, num_iters):
    """
    VALIDATIONCURVE Generate the train and validation errors needed to
    plot a validation curve that we can use to select lambda
    [lambda_vec, error_train, error_val] = ...
        VALIDATIONCURVE(X, y, Xval, yval) returns the train
        and validation errors (in error_train, error_val)
        for different values of lambda. You are given the training set (X,
        y) and validation set (Xval, yval).
    """
    # Selected values of lambda (you should not change this)
    lbd_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    error_train = np.zeros(len(lbd_vec))
    error_val = np.zeros(len(lbd_vec))

    for i in range(len(lbd_vec)):
        lbd = lbd_vec[i]
        theta = np.zeros((X.shape[1], 1))
        theta, _ = trainLinearReg(X, y, theta, alpha, num_iters, lbd)
        error_train[i], _ = linearRegCostFunction(X, y, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return lbd_vec, error_train, error_val
