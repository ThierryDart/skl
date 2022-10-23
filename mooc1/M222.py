# -*- coding: utf-8 -*-
# MOOC FUN
# M222 - Effect of the sample size in cross-validation
# Understand how the different errors are influenced by the number of samples available
#

print("##### M222 - Effect of the sample size in cross-validation #####")

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target
target *= 100  # rescale the target in k$

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()


print("----- Learning curve -----")
# Synthetically reduce the number of samples used to train the predictive model
# and check the training and testing errors.
# We can vary the number of samples in the training set and repeat the experiment.
# Instead of varying a hyperparameter, we vary the number of training samples

# Compute the learning curve for a decision tree
# and vary the proportion of the training set from 10% to 100%.
import numpy as np
train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)
print("train_sizes:",train_sizes)

# Use a ShuffleSplit cross-validation to assess our predictive model.
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=30, test_size=0.2)

# Carry out the experiment
from sklearn.model_selection import learning_curve
results = learning_curve(
    regressor, data, target, train_sizes=train_sizes, cv=cv,
    scoring="neg_mean_absolute_error", n_jobs=2)
train_size, train_scores, test_scores = results[:3]
# Convert the scores into errors
train_errors, test_errors = -train_scores, -test_scores

# Pplot the curve
import matplotlib.pyplot as plt
plt.errorbar(train_size, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Training error")
plt.errorbar(train_size, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="Testing error")
plt.legend()

plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Mean absolute error (k$)")
_ = plt.title("Learning curve for decision tree")
plt.show()

# Training error: we get an error of 0 k$ --> clearly overfitting

# Testing error: the more samples are added into the training set,
# the lower the testing error becomes. Also, we are searching for the plateau of the testing error


# If adding new samples in the training set does not reduce the testing error,
# we might have reach the Bayes error rate using the available model.
# Using a more complex model might be the only possibility to reduce the testing error further.


