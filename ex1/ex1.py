import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ex1_func import plotData, computeCost, gradientDescent

# Read data from ex1data1.txt
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, [0]]
y = data[:, [1]]
m = len(y)

#%% Plot data
plotData(X, y)

#%% Cost and Gradient descent
X = np.c_[np.ones(m), X]    # Add a column of ones to x
theta = np.zeros([2, 1])    # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('Testing the cost function ...')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f' % J)
print('Expected cost value (approx) 32.07')

J = computeCost(X, y, np.array([[-1], [2]]))
print('With theta = [0 ; 0]\nCost computed = %f' % J)
print('Expected cost value (approx) 54.24')

# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303  1.1664')

# Plot the linear fit
plt.plot(X[:, 1], np.dot(X, theta), '-')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of %f\n' % (predict1*10000))
predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of %f\n' % (predict2*10000))

#%% Visualizing J(theta_0, theta_1)
print('Visualizing J(theta_0, theta_1) ...\n')

#Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost(X, y, t)

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
# Surface plot
X, Y = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, J_vals, cmap='jet')
ax.set_xlabel('$\Theta_0$')
ax.set_ylabel('$\Theta_1$')

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.figure()
plt.contour(X, Y, J_vals, np.logspace(-2, 3, 20), cmap='jet')
plt.xlabel('$\Theta_0$')
plt.ylabel('$\Theta_1$')
plt.plot(theta[0], theta[1], 'rx')
