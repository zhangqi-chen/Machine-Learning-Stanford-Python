import numpy as np
import matplotlib.pyplot as plt
from ex2_func import plotData, sigmoid, costFunction, gradientDescent, plotDecisionBoundary, predict

data = np.loadtxt('ex2data1.txt', delimiter=',')
X, y = data[:, :2], data[:, [2]]

#%% Plotting
# We start the exercise by first plotting the data to understand the
# the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(X, y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

#%% Compute Cost and Gradient
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression.

# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X), 1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):\n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24, 0.2, 0.2]]).T
cost, grad = costFunction(test_theta, X, y)

print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:\n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

#%% Optimizing
# Here we are using gradient desent to find theta (instead of fminunc)
# Result might be little different

iterations = 500
alpha = .001
initial_theta = np.array([[-25, 0.2, 0.2]]).T
theta, J_history = gradientDescent(X, y, initial_theta, alpha, iterations)

# Plot the convergence graph
plt.figure()
plt.plot(np.arange(iterations), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Print theta to screen
print('Cost at theta found by fminunc:', cost)
print('Expected cost (approx): 0.203')
print('theta:', theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

#%% Predict and Accuracies
# After learning the parameters, you'll like to use it to predict the outcomes
# on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and
# score 85 on exam 2 will be admitted.
#
# Furthermore, you will compute the training and test set accuracies of
# our model.

# Predict probability for a student with score 45 on exam 1
# and score 85 on exam 2

grade = np.array([1, 45, 85])
prob = sigmoid(grade * theta)

print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %f' % (np.mean(p == y)*100))
print('Expected accuracy (approx): 89.0')
