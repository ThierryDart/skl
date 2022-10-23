# -*- coding: utf-8 -*-
# MOOC FUN
# M531-exo - to find out whether a decision tree model is able to extrapolate.
# By extrapolation, we refer to values predicted by a model outside of the range of feature values seen during the training.

print("##### M531-exo - to find out whether a decision tree model is able to extrapolate. #####")

import pandas as pd

penguins = pd.read_csv("./datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

data_train, target_train = penguins[data_columns], penguins[target_column]


print("----- Linear regression and decision tree regressor 3 -----")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

linear_regression = LinearRegression()
tree = DecisionTreeRegressor(max_depth=3)

linear_regression.fit(data_train, target_train)
tree.fit(data_train, target_train)

print("-----Interpolation -----")

# Create a testing dataset, ranging from the minimum to the maximum of
# the flipper length of the training dataset.
# Get the predictions of each model using this test dataset.

import numpy as np

data_test = pd.DataFrame(np.arange(data_train[data_columns[0]].min(),
                                   data_train[data_columns[0]].max()),
                         columns=data_columns)

target_predicted_linear_regression = linear_regression.predict(data_test)
target_predicted_tree = tree.predict(data_test)

# Create a scatter plot containing the training samples and
# superimpose the predictions of both model on the top.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, target_predicted_linear_regression,
         label="Linear regression")
plt.plot(data_test, target_predicted_tree, label="Decision tree")
plt.legend()
_ = plt.title("Prediction of linear model and a decision tree")

# In some sense, we observe the capabilities of our model to interpolate.


print("----- Extrapolation -----")
# will check the extrapolation capabilities of each model


offset = 30
data_test = pd.DataFrame(np.arange(data_train[data_columns[0]].min() - offset,
                                   data_train[data_columns[0]].max() + offset),
                         columns=data_columns)


target_predicted_linear_regression = linear_regression.predict(data_test)
target_predicted_tree = tree.predict(data_test)


plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, target_predicted_linear_regression,
         label="Linear regression")
plt.plot(data_test, target_predicted_tree, label="Decision tree")
plt.legend()
_ = plt.title("Prediction of linear model and a decision tree")

# Decision trees are non-parametric models and we observe that they cannot extrapolate.

