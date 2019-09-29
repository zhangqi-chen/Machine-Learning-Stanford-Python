import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """
    PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    """
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    plt.figure()
    plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+')
    plt.plot(X[neg][:, 0], X[neg][:, 1], 'yo')


def sigmoid(z):
    """
    SIGMOID Compute sigmoid function
    g = SIGMOID(z) computes the sigmoid of z.
    """
    g = 1/(1+np.exp(-z))
    return g


def costFunction(theta, X, y):
    """
    COSTFUNCTION Compute cost and gradient for logistic regression
    J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    # J=1/m*[-y'*log(h)-(1-y)'*log(1-h)]
    J = (-y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h)))/m
    J = float(J)
    # grad=1/m*(X'*(h-y))
    grad = X.T.dot(h-y)/m
    return J, grad


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = costFunction(theta, X, y)
        theta = theta-alpha*grad
    return theta, J_history


def mapFeature(X1, X2):
    """
    MAPFEATURE Feature mapping function to polynomial features

        MAPFEATURE(X1, X2) maps the two input features
        to quadratic features used in the regularization exercise.

        Returns a new feature array with more features, comprising of
        X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

        Inputs X1, X2 must be the same size
    """
    degree = 6
    out = np.ones((len(X1), 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            out = np.c_[out, X1**(i-j)*X2**j]
    return out


def plotDecisionBoundary(theta, X, y):
    """
    PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    the decision boundary defined by theta
        PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
        positive examples and o for the negative examples. X is assumed to be
        a either
        1) Mx3 matrix, where the first column is an all-ones column for the
           intercept.
        2) MxN, N>3 matrix, where the first column is all-ones
    """
    plotData(X[:, 1:], y)

    if X.shape[1] < 4:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [X[:, 1].min()-2, X[:, 1].max()+2]

        # Calculate the decision boundary line
        plot_y = -(theta[1]*plot_x+theta[0])/theta[2]

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)
    else:
        #Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[[i]], v[[j]]).dot(theta)

        z = z.T     # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        u, v = np.meshgrid(u, v)
        plt.contour(u, v, z, [0])


def predict(theta, X):
    """
    PREDICT Predict whether the label is 0 or 1 using learned logistic
    regression parameters theta
    p = PREDICT(theta, X) computes the predictions for X using a
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    """
    p = sigmoid(X.dot(theta)) >= 0.5
    return p


def costFunctionReg(theta, X, y, lbd):
    """
    COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    # J=1/m*[-y'*log(h)-(1-y)'*log(1-h)]+(lambda/2m)*theta'*theta (ignore theta_0)
    J = (-y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h)))/m
    t = theta[1:]
    J = J+lbd/2/m*(t.T.dot(t))
    J = float(J)
    # grad=1/m*(X'*(h-y)) + theta (ignore theta_0)
    grad = X.T.dot(h-y)/m
    grad[1:] = grad[1:]+lbd/m*t
    return J, grad


def gradientDescentReg(X, y, theta, alpha, lbd, num_iters):
    """
    GRADIENTDESCENTREG Performs gradient descent to learn theta
    theta = GRADIENTDESCENTREG(x, y, theta, alpha, lbd, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = costFunctionReg(theta, X, y, lbd)
        t = theta[1:]
        grad[1:] += lbd/m*t
        theta = theta-alpha*grad
    return theta, J_history
