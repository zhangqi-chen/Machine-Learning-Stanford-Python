import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from ex8_func import cofiCostFunc, checkCostFunction, loadMovieList, normalizeRatings, RatingsGradientDescent

#%% Loading movie ratings dataset
# You will start by loading the movie ratings dataset to understand the
# structure of the data.

print('Loading movie ratings dataset.\n')

# Load data
data = loadmat('ex8_movies.mat')
Y, R = data['Y'], data['R']

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

# From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): %f / 5' % Y[0, R[0, :]].mean())

# We can "visualize" the ratings matrix by plotting it with imshow
plt.figure()
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')

#%% Collaborative Filtering Cost Function
# You will now implement the cost function for collaborative filtering.
# To help you debug your cost function, we have included set of weights
# that we trained on that. Specifically, you should complete the code in
# cofiCostFunc.m to return J.

# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
datap = loadmat('ex8_movieParams.mat')
X, Theta = datap['X'], datap['Theta']
num_users, num_movies, num_features = datap['num_users'], datap['num_movies'], datap['num_features']

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Evaluate cost function
J, _ = cofiCostFunc(np.r_[X.flatten(), Theta.flatten()], Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: %f' % J)
print('(this value should be about 22.22)\n')

#%% Collaborative Filtering Gradient
# Once your cost function matches up with ours, you should now implement
# the collaborative filtering gradient function. Specifically, you should
# complete the code in cofiCostFunc.m to return the grad argument.

print('Checking Gradients (without regularization) ... \n')

# Check gradients by running checkNNGradients
checkCostFunction()

#%% Collaborative Filtering Cost Regularization
# Now, you should implement regularization for the cost function for
# collaborative filtering. You can implement it by adding the cost of
# regularization to the original cost computation.

# Evaluate cost function
J, _ = cofiCostFunc(np.r_[X.flatten(), Theta.flatten()], Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5): %f ' % J)
print('(this value should be about 31.34)\n')

#%% Collaborative Filtering Gradient Regularization
# Once your cost matches up with ours, you should proceed to implement
# regularization for the gradient.

print('Checking Gradients (with regularization) ... \n')

# Check gradients by running checkNNGradients
checkCostFunction(1.5)

#%% Entering ratings for a new user
# Before we will train the collaborative filtering model, we will first
# add ratings that correspond to a new user that we just observed. This
# part of the code will also allow you to put in your own ratings for the
# movies in our dataset!

movieList = loadMovieList()

# Initialize my ratings
my_ratings = np.zeros(1682)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings[6] = 3
my_ratings[11] = 5
my_ratings[53] = 4
my_ratings[63] = 5
my_ratings[65] = 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

print('\nNew user ratings:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList[i]))

#%% Learning Movie Ratings
# Now, you will train the collaborative filtering model on a movie rating
# dataset of 1682 movies and 943 users

print('Training collaborative filtering...\n')

# Load data

Y, R = data['Y'], data['R']
# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

# Add our own ratings to the data matrix
Y = np.c_[my_ratings, Y]
R = np.c_[my_ratings == 0, R]

# Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)

# Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

initial_params = np.r_[X.flatten(), Theta.flatten()]

# Set Regularization
lbd = 10

# Gradient Descent parameters
alpha = 0.0005
num_iters = 500
theta, J_h = RatingsGradientDescent(initial_params, Y, R, num_users, num_movies, num_features, lbd, alpha, num_iters)

plt.figure()
plt.plot(np.arange(num_iters), J_h)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

# Unfold the returned theta back into U and W
X = np.reshape(theta[:num_movies*num_features], (num_movies, num_features))
Theta = np.reshape(theta[num_movies*num_features:], (num_users, num_features))

print('Recommender system learning completed.\n')

#%% Recommendation for you
# After training the model, you can now make recommendations by computing
# the predictions matrix.

p = X.dot(Theta.T)
my_predictions = p[:, 0] + Ymean[:, 0]

movieList = loadMovieList()

r = np.sort(my_predictions)[::-1]
ix = np.argsort(-my_predictions)

print('Top recommendations for you:')
for i in range(10):
    j = ix[i]
    print('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

print('\nOriginal ratings provided:')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s' % (my_ratings[i], movieList[i]))
