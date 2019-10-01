import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex5_func import linearRegCostFunction, trainLinearReg, learningCurve, polyFeatures, featureNormalize, plotFit, validationCurve

#%% Loading and Visualizing Data
# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = loadmat('ex5data1.mat')
X, y = data['X'], data['y']
Xval, yval = data['Xval'], data['yval']
Xtest, ytest = data['Xtest'], data['ytest']

# m = Number of examples
m = X.shape[0]

# Plot training data
plt.figure()
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

#%% Regularized Linear Regression Cost

theta = np.array([[1, 1]]).T
J, _ = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Cost at theta = [1 ; 1]:', J)
print('(this value should be about 303.993192)')

#%% Regularized Linear Regression Gradient

_, grad = linearRegCostFunction(np.c_[np.ones(m), X], y, theta, 1)

print('Gradient at theta = [1 ; 1]:\n', grad)
print('(this value should be about [-15.303016; 598.250744])')

#%% Train Linear Regression
# Here we are still using gradient descent to train LR

# Note: The data is non-linear, so this will not give a great fit.

# Train linear regression with lambda = 0
lbd = 0
alpha = 0.001
num_iters = 500

theta = np.zeros((2, 1))
theta, J_h = trainLinearReg(np.c_[np.ones(m), X], y, theta, alpha, num_iters, lbd)

# Convergence
plt.figure()
plt.plot(np.arange(num_iters), J_h)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Plot fit over the data
plt.figure()
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.c_[np.ones(m), X].dot(theta), '--')

#%% Learning Curve for Linear Regression
# Next, you should implement the learningCurve function.
# Note: Since the model is underfitting the data, we expect to see a graph with "high bias"

lbd = 0
error_train, error_val = learningCurve(np.c_[np.ones(m), X], y, np.c_[np.ones(Xval.shape[0]), Xval], yval, alpha, num_iters, lbd)

plt.figure()
plt.plot(np.arange(m), error_train, label='Train')
plt.plot(np.arange(m), error_val, label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

#%% Feature Mapping for Polynomial Regression
# One solution to this is to use polynomial regression. You should now
# complete polyFeatures to map each example into its powers

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test-mu)/sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val-mu)/sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1:')
print(X_poly[0])

#%% Learning Curve for Polynomial Regression
# Now, you will get to experiment with polynomial regression with multiple
# values of lambda. The code below runs polynomial regression with
# lambda = 0. You should try running the code with different values of
# lambda to see how the fit and learning curve change.

lbd = 0
theta = np.zeros((X_poly.shape[1], 1))
alpha = 0.1
num_iters = 500
theta, _ = trainLinearReg(X_poly, y, theta, alpha, num_iters, lbd)

# Plot training data and fit
plt.figure()
plt.plot(X, y, 'rx')
plotFit(X.min(), X.max(), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %f)' % lbd)

error_train, error_val = learningCurve(np.c_[np.ones(m), X_poly], y, np.c_[np.ones(X_poly_val.shape[0]), X_poly_val], yval, alpha, num_iters, lbd)

plt.figure()
plt.plot(np.arange(m), error_train, label='Train')
plt.plot(np.arange(m), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (lambda = %f)' % lbd)
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('Error')

print('Polynomial Regression (lambda = %f)' % lbd)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

#%% Validation for Selecting Lambda
# You will now implement validationCurve to test various values of
# lambda on a validation set. You will then use this to select the
# "best" lambda value.
alpha = 0.1
num_iters = 500
lbd_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval, alpha, num_iters)

plt.figure()
plt.plot(lbd_vec, error_train, label='Train')
plt.plot(lbd_vec, error_val, label='Cross Validation')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('Error')

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lbd_vec)):
    print(' %f\t%f\t%f' % (lbd_vec[i], error_train[i], error_val[i]))
