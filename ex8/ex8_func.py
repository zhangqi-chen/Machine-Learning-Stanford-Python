import numpy as np
import matplotlib.pyplot as plt


def estimateGaussian(X):
    """
    ESTIMATEGAUSSIAN This function estimates the parameters of a
    Gaussian distribution using the data in X

        [mu sigma2] = estimateGaussian(X),
        The input X is the dataset with each n-dimensional data point in one row
        The output is an n-dimensional vector mu, the mean of the data set
        and the variances sigma^2, an n x 1 vector
    """
    mu = X.mean(0, keepdims=True).T
    sigma2 = X.var(0, keepdims=True).T
    return mu, sigma2


def multivariateGaussian(X, mu, Sigma2):
    """
    MULTIVARIATEGAUSSIAN Computes the probability density function of the
    multivariate gaussian distribution.

        p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
        density function of the examples X under the multivariate gaussian
        distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
        treated as the covariance matrix. If Sigma2 is a vector, it is treated
        as the \sigma^2 values of the variances in each dimension (a diagonal
        covariance matrix)
    """
    k = mu.shape[0]

    if Sigma2.shape[1] == 1 or Sigma2.shape[0] == 1:
        Sigma2 = np.diag(Sigma2[:, 0])

    X = (X-mu.T).copy()
    p = (2*np.pi)**(-k/2)*np.linalg.det(Sigma2)**-0.5
    p = p*np.exp(-0.5*(X.dot(np.linalg.pinv(Sigma2))*X).sum(1, keepdims=True))
    return p


def visualizeFit(X, mu, sigma2):
    """
    VISUALIZEFIT Visualize the dataset and its estimated distribution.

        VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
        probability density function of the Gaussian distribution. Each example
        has a location (x1, x2) that depends on its feature values.
    """
    X1 = np.arange(0, 35.1, .5)
    X2 = np.arange(0, 35.1, .5)
    X1, X2 = np.meshgrid(X1, X2)
    Z = multivariateGaussian(np.c_[X1.flatten(), X2.flatten()], mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'bx')
    # Do not plot if there are infinities
    if np.isinf(Z).sum() == 0:
        plt.contour(X1, X2, Z, 10**np.arange(-20., 0, 3))


def selectThreshold(yval, pval):
    """
    SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting outliers

        [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
        threshold to use for selecting outliers based on the results from a
        validation set (pval) and the ground truth (yval).
    """
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (pval.max()-pval.min())/1000
    for epsilon in np.arange(pval.min(), pval.max()+stepsize/2, stepsize):
        predictions = (pval < epsilon)
        tp = ((predictions == 1) & (yval == 1)).sum()
        fp = ((predictions == 1) & (yval == 0)).sum()
        fn = ((predictions == 0) & (yval == 1)).sum()
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        F1 = 2*prec*rec/(prec+rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lbd):
    """
    COFICOSTFUNC Collaborative filtering cost function

        [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
        num_features, lambda) returns the cost and gradient for the
        collaborative filtering problem.
    """
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features))
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))

    # J=sum((X*Theta'-Y)^2) where R[i,j]==1
    h = X.dot(Theta.T)-Y
    M = h**2
    J = (M*R).sum()/2
    reg = lbd/2*((X**2).sum()+(Theta**2).sum())
    J = J+reg

    X_grad = (h*R).dot(Theta)+lbd*X
    Theta_grad = (h*R).T.dot(X)+lbd*Theta

    grad = np.r_[X_grad.flatten(), Theta_grad.flatten()]
    return J, grad


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


def checkCostFunction(lbd=0):
    """
    CHECKCOSTFUNCTION Creates a collaborative filering problem
    to check your cost function and gradients

        CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem
        to check your cost function and gradients, it will output the
        analytical gradients produced by your code and the numerical gradients
        (computed using computeNumericalGradient). These two gradient
        computations should result in very similar values.
    """
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > .5] = 0
    R = np.zeros(Y.shape)
    R[Y == 0] = 1

    # Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    def Jfunc(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lbd)

    numgrad = computeNumericalGradient(Jfunc, np.r_[X.flatten(), Theta.flatten()])

    cost, grad = cofiCostFunc(np.r_[X.flatten(), Theta.flatten()], Y, R, num_users, num_movies, num_features, lbd)

    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then')
    print('the relative difference will be small (less than 1e-9).')
    print('Relative Difference: %g\n' % diff)


def loadMovieList():
    """
    GETMOVIELIST reads the fixed movie list in movie.txt and returns a
    cell array of the words

        movieList = GETMOVIELIST() reads the fixed movie list in movie.txt
        and returns a cell array of the words in movieList.
    """
    # Read the fixed movieulary list
    fid = open('movie.txt', 'r', encoding='UTF-8')
    ls = fid.readlines()
    fid.close()
    movieList = [i[i.find(' ')+1:-1] for i in ls]
    return movieList


def normalizeRatings(Y, R):
    """
    NORMALIZERATINGS Preprocess data by subtracting mean rating for every movie (every row)

        [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
        has a rating of 0 on average, and returns the mean rating in Ymean.
    """
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros(Y.shape)
    for i in range(m):
        idx = np.where(R[i] == 1)[0]
        Ymean[i, 0] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx]-Ymean[i, 0]
    return Ynorm, Ymean


def RatingsGradientDescent(params, Y, R, num_users, num_movies, num_features, lbd, alpha, num_iters):
    """
    Gradientdescent to learn ratings
    """
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        J_history[i], grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lbd)
        params = params-alpha*grad
        if i % 100 == 99:
            print('Step %i, cost=%f' % (i+1, J_history[i]))
    return params, J_history
