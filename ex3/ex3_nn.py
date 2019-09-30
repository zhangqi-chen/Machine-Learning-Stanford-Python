import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex3_func import displayData, predict

input_layer_size = 400      # 20x20 Input Images of Digits
hidden_layer_size = 25      # 25 hidden units
num_labels = 10             # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

#%% Loading and Visualizing Data

print('Loading and Visualizing Data ...')
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.randint(0, m, 100)
sel = X[rand_indices, :]

displayData(sel)

#%% Loading Pameters
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
pdata = loadmat('ex3weights.mat')
Theta1, Theta2 = pdata['Theta1'], pdata['Theta2']

#%% Implement Predict
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y)*100))

# To give you an idea of the network's output, you can also run
# through the examples one at the a time to see what it is predicting.

# Randomly permute examples
rp = np.random.randint(0, m, m)
for i in range(m):
    # Display
    print('Displaying Example Image')
    displayData(X[[rp[i]], :])

    pred = predict(Theta1, Theta2, X[[rp[i]], :])
    print('Neural Network Prediction: %d (digit %d)\n', pred, np.mod(pred, 10))
    plt.pause(0.5)
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
