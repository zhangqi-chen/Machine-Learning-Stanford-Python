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


def sigmoidGradient(z):
    """
    SIGMOIDGRADIENT returns the gradient of the sigmoid function
    evaluated at z
    g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element.
    """
    g = sigmoid(z)*(1-sigmoid(z))
    return g


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd):
    """
    NNCOSTFUNCTION Implements the neural network cost function for a two layer
    neural network which performs classification

        (J, grad) = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
        X, y, lambda) computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    w1, h1 = hidden_layer_size, input_layer_size+1
    Theta1 = nn_params[:w1*h1].reshape((w1, h1))
    w2, h2 = num_labels, hidden_layer_size+1
    Theta2 = nn_params[w1*h1:].reshape((w2, h2))
    m = X.shape[0]

    # FeedForward
    a1 = np.c_[np.ones(m), X]
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones(m), a2]
    z3 = a2.dot(Theta2.T)
    a3 = sigmoid(z3)
    h = a3

    # y_predict
    yp = np.zeros((m, num_labels))
    for i in range(m):
        yp[i, y[i][0]-1] = 1

    # Cost function with regularzation
    J = (-yp*np.log(h)-(1-yp)*np.log(1-h)).sum()/m
    reg = ((Theta1[:, 1:]**2).sum()+(Theta2[:, 1:]**2).sum())/2/m
    J = J+reg*lbd

    # Backpropagation with regularzation
    # d3=dz3, d2=dz2
    d3 = h-yp
    d2 = (d3.dot(Theta2))[:, 1:]*sigmoidGradient(z2)

    # d Theta calc
    Theta2_grad = d3.T.dot(a2)/m
    Theta1_grad = d2.T.dot(a1)/m

    Theta2_grad[:, 1:] += lbd/m*Theta2[:, 1:]
    Theta1_grad[:, 1:] += lbd/m*Theta1[:, 1:]

    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())

    return J, grad


def randInitializeWeights(L_in, L_out):
    """
    RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
    W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
    of a layer with L_in incoming connections and L_out outgoing
    connections.

    Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1+L_in)*2*epsilon_init-epsilon_init
    return W


def debugInitializeWeights(fan_out, fan_in):
    """
    DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging
    W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
    of a layer with fan_in incoming connections and fan_out outgoing
    connections using a fix set of values

    Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """
    W = np.zeros((fan_out, 1 + fan_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.sin(np.arange(1, W.size+1)).reshape(W.shape)/10
    return W


def computeNumericalGradient(J, theta):
    """
    COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.
    numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.

    Notes: The following code implements numerical gradient checking, and
    returns the numerical gradient.It sets numgrad(i) to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., numgrad(i) should
    be the (approximately) the partial derivative of J with respect
    to theta(i).)
    """
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1 = J(theta - perturb)[0]
        loss2 = J(theta + perturb)[0]
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad


def checkNNGradients(lbd=0):
    """
    CHECKNNGRADIENTS Creates a small neural network to check the
    backpropagation gradients
    CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
    backpropagation gradients, it will output the analytical gradients
    produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should
    result in very similar values.
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = np.mod(np.array([list(range(1, m+1))]).T, num_labels)+1

    # Unroll parameters
    nn_params = np.append(Theta1.flatten(), Theta2.flatten())

    # Short hand for cost function
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %g' % diff)


def nngradientDescent(X, y, nn_params, input_layer_size, hidden_layer_size, num_labels, alpha, lbd, num_iters):
    """
    NNGRADIENTDESCENT Performs gradient descent to learn nn_params in nn
    """
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)
        nn_params = nn_params-alpha*grad
        if i % 100 == 99:
            print('Step %i, cost=%f' % (i+1, J_history[i]))
    return nn_params, J_history


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
