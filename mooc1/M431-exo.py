# -*- coding: utf-8 -*-
# MOOC FUN
# M431-exo - Non linear regression
# to build an intuition on what will be the parameters' values of a linear model
# when the link between the data and the target is non-linear

print("##### M431-exo - Non linear regression #####")


print("----- Generate such non-linear data -----")

import numpy as np
# Set the seed for reproduction
rng = np.random.RandomState(0)

# Generate data
n_sample = 100
data_max, data_min = 1.4, -1.4
len_data = (data_max - data_min)
data = rng.rand(n_sample) * len_data - len_data / 2
noise = rng.randn(n_sample) * .3
target = data ** 3 - 0.5 * data ** 2 + noise

import pandas as pd
import seaborn as sns

full_data = pd.DataFrame({"data": data, "target": target})
_ = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                    alpha=0.5)

print("-----Linear model -----")

def f(data, weight=0, intercept=0):
    target_predict = weight * data + intercept
    return target_predict


predictions = f(data, weight=1.2, intercept=-0.2)

ax = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                     alpha=0.5)
_ = ax.plot(data, predictions)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(target, f(data, weight=1.2, intercept=-0.2))
print(f"The MSE is {error}")


print("-----Train a linear model -----")
# In scikit-learn, by convention data (also called X) should be a 2D matrix
# of shape (n_samples, n_features). If data is a 1D vector, you need to reshape it

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
data_2d = data.reshape(-1, 1)
linear_regression.fit(data_2d, target)

predictions = linear_regression.predict(data_2d)

ax = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                     alpha=0.5)
_ = ax.plot(data, predictions)

error = mean_squared_error(target, predictions)
print(f"The MSE is {error}")




