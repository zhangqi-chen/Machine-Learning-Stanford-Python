import numpy as np
import matplotlib.pyplot as plt
from ex2_func import plotData, mapFeature, costFunctionReg, gradientDescentReg, plotDecisionBoundary, predict


data = np.loadtxt('ex2data2.txt', delimiter=',')
X, y = data[:, :2], data[:, [2]]

plotData(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

#%% Regularized Logistic Regression
#  In this part, you are given a dataset with data points that are not
# linearly separable. However, you would still like to use logistic
# regression to classify the data points.

# To do so, you introduce more features to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial
# regression).

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, [0]], X[:, [1]])

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1
lbd = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, lbd)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1], 1))
cost, grad = costFunctionReg(test_theta, X, y, 10)

print('Cost at test theta (with lambda = 10):', cost)
print('Expected cost (approx): 3.16')
print('Gradient at test theta - first five values only:')
print(grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

#%% Regularization and Accuracies
# Optional Exercise:
# In this part, you will get to try different values of lambda and
# see how regularization affects the decision coundart
#
# Try the following values of lambda (0, 1, 10, 100).
#
# How does the decision boundary change when you vary lambda? How does
# the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

# Set regularization parameter lambda to 1 (you should vary this)
lbd = 1

# Use gradient descent
iterations = 500
alpha = .5

theta, J_history = gradientDescentReg(X, y, initial_theta, alpha, lbd, iterations)

# Plot the convergence graph
plt.figure()
plt.plot(np.arange(iterations), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.title('lambda = %g' % lbd)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f' % (np.mean(p == y)*100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')
