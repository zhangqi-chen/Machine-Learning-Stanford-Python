import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex7_func import findClosestCentroids, computeCentroids, runkMeans, kMeansInitCentroids

#%% Find Closest Centroids

print('Finding closest centroids.')

# Load an example dataset that we will be using
data1 = loadmat('ex7data2.mat')
X = data1['X']

# Select an initial set of centroids
K = 3       # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples:')
print(idx[:3]+1)
print('(the closest centroids should be 1, 3, 2 respectively)')

#%% Compute Means
# After implementing the closest centroids function, you should now
# complete the computeCentroids function.

print('Computing centroids means.')

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:\n', centroids)
print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')

#%% K-Means Clustering
# After you have completed the two functions computeCentroids and
# findClosestCentroids, you have all the necessary pieces to run the
# kMeans algorithm. In this part, you will run the K-Means algorithm on
# the example dataset we have provided.

print('Running K-Means clustering on example dataset.')

# Load an example dataset
data2 = loadmat('ex7data2.mat')
X = data2['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.')

#%% K-Means Clustering on Pixels
# In this exercise, you will use K-Means to compress an image. To do this,
# you will first run K-Means on the colors of the pixels in the image and
# then you will map each pixel onto its closest centroid.

print('Running K-Means clustering on pixels from an image.')

# Load an image of a bird (no need to divided by 255)
A = plt.imread('bird_small.png')

# If imread does not work for you, you can try instead
A = loadmat('bird_small.mat')
A = A['A']

A = A/255     # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = np.reshape(A, (img_size[0]*img_size[1], 3))

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids randomly.
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

#%% Image Compression
# In this part of the exercise, you will use the clusters of K-Means to compress an image.

print('Applying K-Means to compress an image.')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx, :]

# Reshape the recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, (img_size[0], img_size[1], 3))

# Display the original image
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(A)
plt.title('Original')

# Display compressed image side by side
plt.subplot(122)
plt.imshow(X_recovered)
plt.title('Compressed, with %d colors.' % K)
