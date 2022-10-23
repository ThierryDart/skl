# -*- coding: utf-8 -*-
# MOOC FUN
# M21 - The framework and why do we need it
# Go into details into the cross-validation framework
#

print("##### M21 - The framework and why do we need it #####")

from sklearn.datasets import fetch_california_housing

# The aim is to predict the median value of houses in an area in California.
# The target to be predicted is a continuous variable, this task is a regression
housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target

print(housing.DESCR)
print(data.head())

target *= 100
target.head()

print("----- Training error vs testing error -----")

# To solve this regression task, we will use a decision tree regressor.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(data, target)

# Potential statistical performance ?, we use the mean absolute error
from sklearn.metrics import mean_absolute_error
target_predicted = regressor.predict(data)
score = mean_absolute_error(target, target_predicted)
print(f"On average, our regressor makes an error of {score:.2f} k$")

# we trained and predicted on the same dataset !
# This error computed above is called the empirical error or training error.

# We trained a predictive model to minimize the training error
# but our aim is to minimize the error on data that has not been seen during training.
# This error is also called the generalization error or the "true" testing error.


# The most basic evaluation involves:
# - splitting our dataset into two subsets: a training set and a testing set;
# - fitting the model on the training set;
# - estimating the training error on the training set;
# - estimating the testing error on the testing set.

# let's split our dataset.

from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

# let's train our model.
regressor.fit(data_train, target_train)

# Training error.
target_predicted = regressor.predict(data_train)
score = mean_absolute_error(target_train, target_predicted)
print(f"The training error of our model is {score:.2f} k$")

# our model memorized the training set.

# Testing error
target_predicted = regressor.predict(data_test)
score = mean_absolute_error(target_test, target_predicted)
print(f"The testing error of our model is {score:.2f} k$")


print("----- Stability of the cross-validation estimates -----")

# When doing a single train-test split we don't give any indication regarding the robustness of the evaluation of our predictive model
# Cross-validation allows estimating the robustness of a predictive model by repeating the splitting procedure;
# and thus some estimate of the variability of the model statistical performance.
# Different cross-validation strategies: focus on one called "shuffle-split".

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error")

import pandas as pd
cv_results = pd.DataFrame(cv_results)
print(cv_results.head())

# A score is a metric for which higher values mean better results.
# all error metrics in scikit-learn can be transformed into a score to be used in cross_validate.
# To do so, you need to pass a string of the error metric with instance scoring="neg_mean_absolute_error".

# revert the negation to get the actual error
cv_results["test_error"] = -cv_results["test_score"]
print(cv_results.head(10))
print(len(cv_results))

import matplotlib.pyplot as plt

cv_results["test_error"].plot.hist(bins=10, edgecolor="black", density=True)
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Test error distribution")
plt.show()

print(f"The mean cross-validated testing error is: "
      f"{cv_results['test_error'].mean():.2f} k$")

print(f"The standard deviation of the testing error is: "
      f"{cv_results['test_error'].std():.2f} k$")

# Our cross-validation estimate of the testing error is 46.36 +/- 1.17 k$.

# Plot the distribution of the target variable
target.plot.hist(bins=20, edgecolor="black", density=True)
plt.xlabel("Median House Value (k$)")
_ = plt.title("Target distribution")
plt.show()

print(f"The mean cross-validated of the target is: {target.mean():.2f} k$")
print(f"The standard deviation of the target is: {target.std():.2f} k$")

# The mean estimate of the testing error obtained by cross-validation
# is a bit smaller than the natural scale of variation of the target variable.
# Furthermore, the standard deviation of the cross validation estimate of the testing error is even smaller.

print("----- More detail regarding cross_validate -----")
# During cross-validation, many models are trained and evaluated
# Retrieve theses fitted models for each of the splits/folds
# by passing the option return_estimator=True in cross_validate

cv_results = cross_validate(regressor, data, target, return_estimator=True)
print("cv_results:\n", cv_results)
print("estimator:\n", cv_results["estimator"])

# Interested in the test score: a cross_val_score function
# Identical to calling the cross_validate function and to select the test_score only

from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, data, target)
print("Scores:\n",scores)
