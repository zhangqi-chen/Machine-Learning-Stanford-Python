import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex4_func import displayData, nnCostFunction, sigmoidGradient, randInitializeWeights, checkNNGradients, nngradientDescent, predict

# Setup the parameters you will use for this exercise
input_layer_size = 400     # 20x20 Input Images of Digits
hidden_layer_size = 25      # 25 hidden units
num_labels = 10             # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10

#%% Loading and Visualizing Data
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('ex4data1.mat')
X, y = data['X'], data['y']
m = X.shape[0]


# Randomly select 100 data points to display
sel = np.random.randint(0, m, 100)

displayData(X[sel, :])

#%% Loading Parameters
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
pdata = loadmat('ex4weights.mat')
Theta1, Theta2 = pdata['Theta1'], pdata['Theta2']

# Unroll parameters
nn_params = np.append(Theta1.flatten(), Theta2.flatten())

#%% Compute Cost (Feedforward)
# To the neural network, you should first start by implementing the
# feedforward part of the neural network that returns the cost only. You
# should complete the code in nnCostFunction.m to return cost. After
# implementing the feedforward to compute the cost, you can verify that
# your implementation is correct by verifying that you get the same cost
# as us for the fixed debugging parameters.
#
# We suggest implementing the feedforward cost *without* regularization
# first so that it will be easier for you to debug. Later, in part 4, you
# will get to implement the regularized cost.

print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lbd = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)

print('Cost at parameters (loaded from ex4weights):', J)
print('this value should be about 0.287629')

#%% Implement Regularization
# Once your cost function implementation is correct, you should now
# continue to implement the regularization with the cost.

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lbd = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)

print('Cost at parameters (loaded from ex4weights):', J)
print('this value should be about 0.383770)')

#%% Sigmoid Gradient
# Before you start implementing the neural network, you will first
# implement the gradient for the sigmoid function. You should complete the
# code in the sigmoidGradient.m file.

print('Evaluating sigmoid gradient...')

g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(g)

#%% Initializing Pameters
# In this part of the exercise, you will be starting to implment a two
# layer neural network that classifies digits. You will start by
# implementing a function to initialize the weights of the neural network
# (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())

#%% Implement Backpropagation
# Once your cost matches up with ours, you should proceed to implement the
# backpropagation algorithm for the neural network. You should add to the
# code you've written in nnCostFunction.m to return the partial
# derivatives of the parameters.

print('Checking Backpropagation...')

# Check gradients by running checkNNGradients
checkNNGradients()

#%% Implement Regularization
# Once your backpropagation implementation is correct, you should now
# continue to implement the regularization with the cost and gradient.

print('Checking Backpropagation (w/ Regularization) ...')

# Check gradients by running checkNNGradients
lbd = 3
checkNNGradients(lbd)

# Also output the costFunction debugging values
debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbd)

print('Cost at (fixed) debugging parameters (w/ lambda = %f): %f' % (lbd, debug_J))
print('(for lambda = 3, this value should be about 0.576051)')

#%% Training NN
# You have now implemented all the code necessary to train a neural network.
# Here we are still using gradient descent

print('Training Neural Network (Gradient Descent) ...')

lbd = 1
alpha = 2
num_iters = 500

nn_params, J_h = nngradientDescent(X, y, initial_nn_params, input_layer_size, hidden_layer_size, num_labels, alpha, lbd, num_iters)

plt.figure()
plt.plot(np.arange(num_iters), J_h)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Obtain Theta1 and Theta2 back from nn_params
w1, h1 = hidden_layer_size, input_layer_size+1
Theta1 = nn_params[:w1*h1].reshape((w1, h1))
w2, h2 = num_labels, hidden_layer_size+1
Theta2 = nn_params[w1*h1:].reshape((w2, h2))

#%% Visualize Weights
# You can now "visualize" what the neural network is learning by
# displaying the hidden units to see what features they are capturing in
# the data.

print('Visualizing Neural Network...')

displayData(Theta1[:, 1:])

#%% Implement Predict
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y)*100))
