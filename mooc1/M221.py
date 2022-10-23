# -*- coding: utf-8 -*-
# MOOC FUN
# M221 - Overfit-generalization-underfit
# Put these two errors into perspective and show how they can help us know
# if our model generalizes, overfit, or underfit.
#

print("##### M221 - Overfit-generalization-underfit #####")

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
# Split data(features) and target
data, target = housing.data, housing.target
target *= 100  # rescale the target in k$


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

print("----- Overfitting vs. underfitting -----")
# To better understand the statistical performance of our model,
# We will compare the testing error with the training error.
# Thus, we'll compute the error on the training set, using cross_validate

import pandas as pd
from sklearn.model_selection import cross_validate, ShuffleSplit

cv = ShuffleSplit(n_splits=30, test_size=0.2)
cv_results = cross_validate(regressor, data, target,
                            cv=cv, scoring="neg_mean_absolute_error",
                            return_train_score=True, n_jobs=2)
cv_results = pd.DataFrame(cv_results)

# Transform the negative mean absolute error into a positive mean absolute error.
scores = pd.DataFrame()
scores[["train error", "test error"]] = -cv_results[["train_score", "test_score"]]

# Plot
import matplotlib.pyplot as plt

scores.plot.hist(bins=50, edgecolor="black", density=True)
plt.xlabel("Mean absolute error (k$)")
_ = plt.title("Train and test errors distribution via cross-validation")
plt.show()

# Small training error (actually zero), meaning that the model is not under-fitting:
# it is flexible enough to capture any variations present in the training set.

# However the significantly larger testing error tells us that the model is over-fitting:
# the model has memorized many variations

print("----- Validation curve -----")
# For the decision tree, the max_depth parameter is used to control
# the tradeoff between under-fitting and over-fitting.

from sklearn.model_selection import validation_curve

max_depth = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    regressor, data, target, param_name="max_depth", param_range=max_depth,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
train_errors, test_errors = -train_scores, -test_scores

plt.plot(max_depth, train_errors.mean(axis=1), label="Training error")
plt.plot(max_depth, test_errors.mean(axis=1), label="Testing error")
plt.legend()

plt.xlabel("Maximum depth of decision tree")
plt.ylabel("Mean absolute error (k$)")
_ = plt.title("Validation curve for decision tree")
plt.show()

# The validation curve can be divided into three areas:
#
# max_depth < 10, the decision tree underfits
# The training error and therefore the testing error are both high
#
# Around max_depth = 10 corresponds to the parameter for which the decision tree generalizes the best.
#
# max_depth > 10, the decision tree overfits.
# The training error becomes very small, while the testing error increases.


# Be aware that looking at the mean errors is quite limiting.
# We should also look at the standard deviation to assess
# the dispersion of the score.
plt.errorbar(max_depth, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label='Training error')
plt.errorbar(max_depth, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label='Testing error')
plt.legend()

plt.xlabel("Maximum depth of decision tree")
plt.ylabel("Mean absolute error (k$)")
_ = plt.title("Validation curve for decision tree")
plt.show()



