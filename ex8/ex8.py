import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex8_func import estimateGaussian, multivariateGaussian, visualizeFit, selectThreshold

#%% Load Example Dataset
# We start this exercise by using a small dataset that is easy to visualize.

# Our example case consists of 2 network server statistics across
# several machines: the latency and throughput of each machine.
# This exercise will help us find possibly faulty (or very fast) machines.

print('Visualizing example dataset for outlier detection.')

# The following command loads the dataset. You should now have the
# variables X, Xval, yval in your environment
data1 = loadmat('ex8data1.mat')
X, Xval, yval = data1['X'], data1['Xval'], data1['yval']

# Visualize the example dataset
plt.figure()
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

#%% Estimate the dataset statistics
# For this exercise, we assume a Gaussian distribution for the dataset.

# We first estimate the parameters of our assumed Gaussian distribution,
# then compute the probabilities for each of the points and then visualize
# both the overall distribution and where each of the points falls in
# terms of that distribution.

print('\nVisualizing Gaussian fit.')

# Estimate my and sigma2
mu, sigma2 = estimateGaussian(X)

# Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2)

# Visualize the fit
visualizeFit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

#%% Find Outliers
# Now you will find a good epsilon threshold using a cross-validation set
# probabilities given the estimated Gaussian distribution

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)
print('\nBest epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 8.99e-05)')
print('   (you should see a Best F1 value of  0.875000)')

# Find the outliers in the training set and plot the
outliers = np.where(p < epsilon)[0]

# Draw a red circle around those outliers
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, fillstyle='none')

#%% Multidimensional Outliers
# We will now use the code from the previous part and apply it to a
# harder problem in which more features describe each datapoint and only
# some features indicate whether a point is an outlier.

# Loads the second dataset. You should now have the
# variables X, Xval, yval in your environment
data2 = loadmat('ex8data2.mat')
X, Xval, yval = data2['X'], data2['Xval'], data2['yval']

# Apply the same steps to the larger dataset
mu, sigma2 = estimateGaussian(X)

# Training set
p = multivariateGaussian(X, mu, sigma2)

# Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2)

# Find the best threshold
epsilon, F1 = selectThreshold(yval, pval)

print('\nBest epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set:  %f' % F1)
print('   (you should see a value epsilon of about 1.38e-18)')
print('   (you should see a Best F1 value of 0.615385)')
print('# Outliers found: %d' % sum(p < epsilon))
