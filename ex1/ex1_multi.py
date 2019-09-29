import numpy as np
import matplotlib.pyplot as plt
from ex1_func import featureNormalize, gradientDescentMulti, normalEqn

#%% Feature Normalization

# Load data
print('Loading data ...')
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, [2]]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset:')
print(' x = \n', X[1:10, :], '\ny = \n', y[1:10, :])

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.c_[np.ones(m), X]

#%% Gradient Descent
# Choose some alpha value
alpha = 0.1
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(np.arange(num_iters), J_history, 'b-')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]]).dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n$%f' % price[0])

#%% Normal Equations
print('Solving with normal equations...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, [2]]
m = len(y)

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), 1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:\n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650, 3]).dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n$%f' % price[0])
