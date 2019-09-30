import numpy as np
import matplotlib.pyplot as plt


def displayData(X, example_width=None):
    """
    DISPLAYDATA Display 2D data in a nice grid
    [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    stored in X in a nice grid. It returns the figure handle h and the
    displayed array if requested.
    """
    if example_width is None:
        example_width = round(np.sqrt(X.shape[1])).astype(int)

    # Compute rows, cols
    m, n = X.shape
    example_height = n//example_width

    # Compute number of items to display
    display_rows = np.floor(np.sqrt(m)).astype(int)
    display_cols = np.ceil(m/display_rows).astype(int)

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            imvalue = X[curr_ex, :].reshape((example_height, example_width))/max_val
            hl = pad+j*(example_height+pad)
            wl = pad+i*(example_width+pad)
            display_array[hl:hl+example_height, wl:wl+example_width] = imvalue
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

    # Display the image
    # numpy.reshape is different from matlab, so transpose is required
    # Otherwise, use order='F' in numpy.reshape
    plt.figure()
    plt.imshow(display_array.T, cmap='gray')
    plt.axis('off')


def sigmoid(z):
    """
    SIGMOID Compute sigmoid function
    g = SIGMOID(z) computes the sigmoid of z.
    """
    g = 1/(1+np.exp(-z))
    return g


def lrCostFunction(theta, X, y, lbd):
    """
    LRCOSTFUNCTION Compute cost and gradient for logistic regression with
    regularization
    J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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


def lrgradientDescent(X, y, theta, alpha, lbd, num_iters):
    """
    LRGRADIENTDESCENT Performs gradient descent to learn theta
    theta = LRGRADIENTDESCENT(x, y, theta, alpha, lbd, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = lrCostFunction(theta, X, y, lbd)
        t = theta[1:]
        grad[1:] += lbd/m*t
        theta = theta-alpha*grad
    return theta, J_history


def oneVsAll(X, y, num_labels, lbd):
    """
    ONEVSALL trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i
    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    logistic regression classifiers and returns each of these classifiers
    in a matrix all_theta, where the i-th row of all_theta corresponds
    to the classifier for label i
    """
    m, n = X.shape
    all_theta = np.zeros((num_labels, n+1))

    X = np.c_[np.ones(m), X]     # Add ones to the X data matrix
    initial_theta = np.zeros((n + 1, 1))

    # Using gradient descent to find theta
    alpha = 0.5
    num_iters = 500
    for c in range(num_labels):
        y_c = (y == c+1)
        theta_c, J_h = lrgradientDescent(X, y_c, initial_theta, alpha, lbd, num_iters)
        all_theta[c] = theta_c.T[0]

        # Plots to track the convergence
#        plt.figure()
#        plt.title('digit=%i'%c)
#        plt.plot(np.arange(num_iters),J_h)
#        plt.xlabel('Number of iterations')
#        plt.ylabel('Cost J')

    return all_theta


def predictOneVsAll(all_theta, X):
    """
    PREDICT Predict the label for a trained one-vs-all classifier. The labels
    are in the range 1..K, where K = size(all_theta, 1).
    p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    for each example in the matrix X. Note that X contains the examples in
    rows. all_theta is a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class. You should set p to a vector
    of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    for 4 examples)
    """
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    predict = sigmoid(X.dot(all_theta.T))
    p = predict.argmax(axis=1)+1
    p = np.r_[[p]].T
    return p


def predict(Theta1, Theta2, X):
    """
    PREDICT Predict the label of an input given a trained neural network
    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    t1 = sigmoid(X.dot(Theta1.T))
    t1 = np.c_[np.ones(m), t1]
    t2 = sigmoid(t1.dot(Theta2.T))
    p = t2.argmax(axis=1)+1
    p = np.r_[[p]].T
    return p
