import numpy as np
import matplotlib.pyplot as plt


def findClosestCentroids(X, centroids):
    """
    FINDCLOSESTCENTROIDS computes the centroid memberships for every example.

        idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
        in idx for a dataset X where each row is a single example. idx = m x 1
        vector of centroid assignments (i.e. each entry in range [1..K])
    """
    idx = np.zeros((X.shape[0], 1)).astype(int)
    for i in range(X.shape[0]):
        idx[i, 0] = ((centroids-X[i])**2).sum(1).argmin()
    return idx


def computeCentroids(X, idx, K):
    """
    COMPUTECENTROIDS returns the new centroids by computing the means of the
    data points assigned to each centroid.

        centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by
        computing the means of the data points assigned to each centroid. It is
        given a dataset X where each row is a single data point, a vector
        idx of centroid assignments (i.e. each entry in range [1..K]) for each
        example, and K, the number of centroids. You should return a matrix
        centroids, where each row of centroids is the mean of the data points
        assigned to it.
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    for c in range(K):
        ip = np.where(idx == c)[0]
        centroids[c] = X[ip].mean(0)
    return centroids


def plotDataPoints(X, idx, K):
    """
    PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    index assignments in idx have the same color.
    """
    # Create palette
    colors = ['r', 'b', 'g', 'y', 'm', 'c']

    # Plot the data
    for i in range(K):
        ip = np.where(idx == i)[0]
        plt.scatter(X[ip, 0], X[ip, 1], c=colors[i % len(colors)])


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    """
    PLOTPROGRESSKMEANS is a helper function that displays the progress of
    k-Means as it is running. It is intended for use only with 2D data.

        PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
        points with colors assigned to each centroid. With the previous
        centroids, it also plots a line between the previous locations and
        current locations of the centroids.
    """
    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx')

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        x1, x2 = centroids[j], previous[j]
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], 'k-')

    # Title
    plt.title('Iteration number %d' % i)


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    is a single example.

        [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
        plot_progress) runs the K-Means algorithm on data matrix X, where each
        row of X is a single example. It uses initial_centroids used as the
        initial centroids. max_iters specifies the total number of interactions
        of K-Means to execute. plot_progress is a true/false flag that
        indicates if the function should also plot its progress as the
        learning happens. This is set to false by default. runkMeans returns
        centroids, a Kxn matrix of the computed centroids and idx, a m x 1
        vector of centroid assignments (i.e. each entry in range [1..K])
    """
    if plot_progress:
        plt.figure()

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    previous_centroids = centroids.copy()
    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print('K-Means iteration %d/%d...' % (i+1, max_iters))

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids.copy()
            plt.pause(0.1)

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    return centroids, idx


def kMeansInitCentroids(X, K):
    """
    KMEANSINITCENTROIDS This function initializes K centroids that are to be
    used in K-Means on the dataset X
    """
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    centroids = X[randidx[:K], :]
    return centroids


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


def pca(X):
    """
    PCA Run principal component analysis on the dataset X

        [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
        Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    m, n = X.shape
    Sigma = X.T.dot(X)/m
    U, S, V = np.linalg.svd(Sigma)
    return U, S


def projectData(X, U, K):
    """
    PROJECTDATA Computes the reduced data representation when projecting only
    on to the top k eigenvectors

        Z = projectData(X, U, K) computes the projection of
        the normalized inputs X into the reduced dimensional space spanned by
        the first K columns of U. It returns the projected examples in Z.
    """
    Ur = U[:, :K]
    Z = X.dot(Ur)
    return Z


def recoverData(Z, U, K):
    """
    RECOVERDATA Recovers an approximation of the original data when using the projected data

        X_rec = RECOVERDATA(Z, U, K) recovers an approximation the
        original data that has been reduced to K dimensions. It returns the
        approximate reconstruction in X_rec.
    """
    Ur = U[:, :K]
    X_rec = Z.dot(Ur.T)
    return X_rec


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
    plt.imshow(display_array.T, cmap='gray')
    plt.axis('off')
