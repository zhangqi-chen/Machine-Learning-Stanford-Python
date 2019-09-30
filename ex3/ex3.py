import numpy as np
from scipy.io import loadmat
from ex3_func import displayData, lrCostFunction, oneVsAll, predictOneVsAll

#%% Loading and Visualizing Data
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400      # 20x20 Input Images of Digits
num_labels = 10             # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

# Load Training Data
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.randint(0, m, 100)
sel = X[rand_indices, :]

displayData(sel)

#%% Vectorize Logistic Regression
# In this part of the exercise, you will reuse your logistic regression
# code from the last exercise. You task here is to make sure that your
# regularized logistic regression implementation is vectorized. After
# that, you will implement one-vs-all classification for the handwritten
# digit dataset.

# Test case for lrCostFunction
print('Testing lrCostFunction() with regularization')

theta_t = np.array([[-2, -1, 1, 2]]).T
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((5, 3), order='F')/10]
y_t = np.array([[1, 0, 1, 0, 1]]).T >= 0.5
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost:', J)
print('Expected cost: 2.534819')
print('Gradients:')
print(grad)
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

#%% One-vs-All Training
print('Training One-vs-All Logistic Regression...')

lbd = 0.1
all_theta = oneVsAll(X, y, num_labels, lbd)

#%% Predict for One-Vs-All
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y)*100))
