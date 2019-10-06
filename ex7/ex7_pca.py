import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from ex7_func import featureNormalize, pca, projectData, recoverData, displayData, \
    kMeansInitCentroids, runkMeans, plotDataPoints

#%% Load Example Dataset
# We start this exercise by using a small dataset that is easily to visualize

print('Visualizing example dataset for PCA.')

# The following command loads the dataset. You should now have the
# variable X in your environment
data1 = loadmat('ex7data1.mat')
X = data1['X']

# Visualize the example dataset
plt.figure()
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis('equal')

#%% Principal Component Analysis
# You should now implement PCA, a dimension reduction technique. You
# should complete the code in pca.m

print('Running PCA on example dataset.')

# Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S = pca(X_norm)

# Compute mu, the mean of the each feature

# Draw the eigenvectors centered at mean of data. These lines show the
# directions of maximum variations in the dataset.
x1, y1 = mu[0]
x2, y2 = mu[0]+1.5*S[0]*U[:, 0].T
x3, y3 = mu[0]+1.5*S[1]*U[:, 1].T
plt.plot([x3, x1, x2], [y3, y1, y2], 'k-')

print('Top eigenvector:')
print(' U(:,1) = %f %f' % (U[0, 0], U[1, 0]))
print('(you should expect to see -0.707107 -0.707107)')

#%% Dimension Reduction
# You should now implement the projection step to map the data onto the
# first k eigenvectors. The code will then plot the data in this reduced
# dimensional space.  This will show you what the data looks like when
# using only the corresponding eigenvectors to reconstruct it.

print('Dimension reduction on example dataset.')

# Plot the normalized dataset (returned from pca)
plt.figure()
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis('equal')

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example:', Z[0])
print('(this value should be about 1.481274)')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)')

# Draw lines connecting the projected points to the original points
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(X_norm.shape[0]):
    x1, y1 = X_norm[i, :]
    x2, y2 = X_rec[i, :]
    plt.plot([x1, x2], [y1, y2], 'k--')

#%% Loading and Visualizing Face Data
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment

print('\nLoading face dataset.')

# Load Face dataset
data2 = loadmat('ex7faces.mat')
X = data2['X']

# Display the first 100 faces in the dataset
plt.figure()
displayData(X[:100, :])

#%% PCA on Face Data: Eigenfaces
# Run PCA and visualize the eigenvectors which are in this case eigenfaces
# We display the first 36 eigenfaces.

print('\nRunning PCA on face dataset.\n(this might take a minute or two ...)')

# Before running PCA, it is important to first normalize X by subtracting
# the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S = pca(X_norm)

# Visualize the top 36 eigenvectors found
plt.figure()
displayData(U[:, :36].T)

#%% Dimension Reduction for Faces
# Project images to the eigen space using the top k eigenvectors
# If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset.')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ', Z.shape)

#%% Visualization of Faces after PCA Dimension Reduction
# Project images to the eigen space using the top K eigen vectors and
# visualize only using those K dimensions
# Compare to the original input, which is also displayed

print('\nVisualizing the projected (reduced dimension) faces.')

K = 100
X_rec = recoverData(Z, U, K)

# Display normalized data
plt.figure(figsize=(10, 5))
plt.subplot(121)
displayData(X_norm[:100, :])
plt.title('Original faces')
plt.axis('equal')

# Display reconstructed data from only k eigenfaces
plt.subplot(122)
displayData(X_rec[:100, :])
plt.title('Recovered faces')
plt.axis('equal')

#%% Optional (ungraded) Exercise: PCA for Visualization
# One useful application of PCA is to use it to visualize high-dimensional
# data. In the last K-Means exercise you ran K-Means on 3-dimensional
# pixel colors of an image. We first visualize this output in 3D, and then
# apply PCA to obtain a visualization in 2D.

# Reload the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = plt.imread('bird_small.png')

# If imread does not work for you, you can try instead
A = loadmat('bird_small.mat')
A = A['A']

A = A/255
img_size = A.shape
X = np.reshape(A, (img_size[0]*img_size[1], 3))
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

# Sample 1000 random indexes (since working with all the data is
# too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.rand(1000, 1)*X.shape[0]).astype(int)[:, 0]

# Setup Color Palette
palette = ['r', 'b', 'g', 'y', 'm', 'c']
colors = [palette[idx[sel[i], 0] % len(palette)] for i in range(len(sel))]

# Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

#%%  Optional (ungraded) Exercise: PCA for Visualization
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.figure()
plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
