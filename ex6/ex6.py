import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex6_func import plotData, linearKernel, gaussianKernel, svmTrain, visualizeBoundaryLinear

#%% Loading and Visualizing Data

print('Loading and Visualizing Data ...')

data=loadmat('ex6data1.mat')
X,y=data['X'],data['y']
y=y.astype(int)

# Plot training data
plotData(X, y)

#%% Training Linear SVM
# The following code will train a linear SVM on the dataset and plot the
# decision boundary learned.

print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svmTrain(X, y, C, linearKernel, 1e-3, 20)
visualizeBoundaryLinear(X, y, model)