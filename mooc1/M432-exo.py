# -*- coding: utf-8 -*-
# MOOC FUN
# M432-exo - Train a linear regression algorithm on a dataset with more than a single feature.
#

print("##### M432-exo - Train a linear regression algorithm on a dataset with more than a single feature. #####")

from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$
print(data.head())

print("----- Linear regression")

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()


print("----- Cross-validation with 10 folds and use the mean absolute error (MAE)")

from sklearn.model_selection import cross_validate

cv_results = cross_validate(linear_regression, data, target,
                            scoring="neg_mean_absolute_error",
                            return_estimator=True, cv=10, n_jobs=2)


print("----- Compute mean and std of the MAE in thousands of dollars (k$)")

print(f"Mean absolute error on testing set: "
      f"{-cv_results['test_score'].mean():.3f} k$ +/- "
      f"{cv_results['test_score'].std():.3f}")


print("----- Show the values of the coefficients for each feature using a boxplot")

import pandas as pd

weights = pd.DataFrame(
    [est.coef_ for est in cv_results["estimator"]], columns=data.columns)


import matplotlib.pyplot as plt

color = {"whiskers": "black", "medians": "black", "caps": "black"}
weights.plot.box(color=color, vert=False)
_ = plt.title("Value of linear regression coefficients")

