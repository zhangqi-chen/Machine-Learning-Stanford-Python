import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    """
    PLOTDATA Plots the data points X and y into a new figure
    PLOTDATA(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.

    Note: This was slightly modified such that it expects y = 1 or y = 0
    """
    # Find Indices of Positive and Negative Examples
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    # Plot Examples
    plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'ko')


def linearKernel(x1, x2):
    """
    LINEARKERNEL returns a linear kernel between x1 and x2
    sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    and returns the value in sim
    """
    sim = x1.T.dot(x2)
    return sim


def gaussianKernel(x1, x2, sigma):
    """
    RBFKERNEL returns a radial basis function kernel between x1 and x2
    sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """
    sim = np.exp(-((x1-x2)**2).sum()/sigma**2/2)
    return sim


class Model:
    """
    class to save model's data
    """

    def __init__(self):
        self.X = None
        self.y = None
        self.kernelFunction = None
        self.b = None
        self.alphas = None
        self.w = None


def svmTrain(X, Y, C, kernelFunction, tol=1e-3, max_passes=5):
    """
    SVMTRAIN Trains an SVM classifier using a simplified version of the SMO algorithm.
        [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
        SVM classifier and returns trained model. X is the matrix of training
        examples.  Each row is a training example, and the jth column holds the
        jth feature.  Y is a column matrix containing 1 for positive examples
        and 0 for negative examples.  C is the standard SVM regularization
        parameter.  tol is a tolerance value used for determining equality of
        floating point numbers. max_passes controls the number of iterations
        over the dataset (without changes to alpha) before the algorithm quits.

    Note: This is a simplified version of the SMO algorithm for training
        SVMs. In practice, if you want to train an SVM classifier, we
        recommend using an optimized package such as:

            LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
            SVMLight (http://svmlight.joachims.org/)
    """
    # Data parameters
    m, n = X.shape

    # Map 0 to -1
    Y = Y.copy()
    Y[Y == 0] = -1

    # Variables
    alphas = np.zeros((m, 1))
    b = 0
    E = np.zeros((m, 1))
    passes = 0
    eta = 0
    L = 0
    H = 0

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)

    # We have implemented optimized vectorized version of the Kernels here so
    # that the svm training will run faster.
    if kernelFunction.__name__ == 'linearKernel':
        # Vectorized computation for the Linear Kernel
        # This is equivalent to computing the kernel on every pair of examples
        K = X.dot(X.T)
    elif 'gaussianKernel' in kernelFunction.__name__:
        # Vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        X2 = (X**2).sum(1)
        K = X2+(X2.T-2*X.dot(X.T))
        K = kernelFunction(1, 0)**K
    else:
        # Pre-compute the Kernel Matrix
        # The following can be slow due to the lack of vectorization
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = kernelFunction(X[i, :].T, X[j, :].T)
                K[j, i] = K[i, j]   # the matrix is symmetric

    # Train
    print('Training ...')
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # Calculate Ei = f(x[i]) - Y[i,0] using (2)
            # E[i,0] = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y[i,0]
            E[i,0] = b + (alphas*Y*K[:, i]).sum() - Y[i,0]

            if (Y[i,0]*E[i,0] < -tol and alphas[i,0] < C) or (Y[i,0]*E[i,0] > tol and alphas[i,0] > 0):

                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = int(m*np.random.rand())
                while j == i:  # Make sure i \neq j
                    j = int(m*np.random.rand())

                # Calculate Ej = f(x[j]) - Y[j,0] using (2).
                E[j,0] = b + (alphas*Y*K[:, j]).sum() - Y[j,0]

                # Save old alphas
                alpha_i_old = alphas[i,0]
                alpha_j_old = alphas[j,0]

                # Compute L and H by (10) or (11).
                if Y[i,0] == Y[j,0]:
                    L = max(0, alphas[j,0] + alphas[i,0] - C)
                    H = min(C, alphas[j,0] + alphas[i,0])
                else:
                    L = max(0, alphas[j,0] - alphas[i,0])
                    H = min(C, C + alphas[j,0] - alphas[i,0])

                if L == H:
                    # continue to next i.
                    continue

                # Compute eta by (14).
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    # continue to next i.
                    continue

                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j,0] = alphas[j,0] - (Y[j,0] * (E[i,0] - E[j,0])) / eta

                # Clip
                alphas[j,0] = min(H, alphas[j,0])
                alphas[j,0] = max(L, alphas[j,0])

                # Check if change in alpha is significant
                if abs(alphas[j,0] - alpha_j_old) < tol:
                    # continue to next i.
                    # replace anyway
                    alphas[j,0] = alpha_j_old
                    continue

                # Determine value for alpha i using (16).
                alphas[i,0] = alphas[i,0] + Y[i,0]*Y[j,0]*(alpha_j_old - alphas[j,0])

                # Compute b1 and b2 using (17) and (18) respectively.
                b1 = b-E[i,0]-Y[i,0]*(alphas[i,0]-alpha_i_old)*K[i, j].T-Y[j,0]*(alphas[j,0]-alpha_j_old)*K[i, j].T
                b2 = b-E[j,0]-Y[i,0]*(alphas[i,0]-alpha_i_old)*K[i, j].T-Y[j,0]*(alphas[j,0]-alpha_j_old)*K[j, j].T

                # Compute b by (19).
                if 0 < alphas[i,0] and alphas[i,0] < C:
                    b = b1
                elif 0 < alphas[j,0] and alphas[j,0] < C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0

        dots = dots + 1
        if dots > 78:
            dots = 0
            print('.')

    print('Done!')

    # Save the model
    idx = np.where(alphas > 0)[0]
    model = Model()
    model.X = X[idx, :]
    model.y = Y[idx]
    model.kernelFunction = kernelFunction
    model.b = b
    model.alphas = alphas[idx]
    model.w = (alphas*Y).T.dot(X).T

    return model


def visualizeBoundaryLinear(X, y, model):
    """
    VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the SVM
    VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary
    learned by the SVM and overlays the data on it
    """
    w = model.w
    b = model.b
    xp = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yp = - (w[0]*xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, 'b-')
