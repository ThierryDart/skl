# -*- coding: utf-8 -*-
# MOOC FUN
# M531 - Decision tree for regression
#



print("##### M531 - Decision tree for regression #####")

import pandas as pd

penguins = pd.read_csv("./datasets/penguins_regression.csv")

data_columns = ["Flipper Length (mm)"]
target_column = "Body Mass (g)"

data_train, target_train = penguins[data_columns], penguins[target_column]

# Create a synthetic dataset containing all possible flipper length
# from the minimum to the maximum of the original data.

import numpy as np

data_test = pd.DataFrame(np.arange(data_train[data_columns[0]].min(),
                                   data_train[data_columns[0]].max()),
                         columns=data_columns)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
_ = plt.title("Illustration of the regression dataset used")


print("----- Linear regression -----")

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(data_train, target_train)
target_predicted = linear_model.predict(data_test)

plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, target_predicted, label="Linear regression")
plt.legend()
_ = plt.title("Prediction function using a LinearRegression")

# We see that a non-regularized LinearRegression is able to fit the data

plt.figure()
ax = sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                     color="black", alpha=0.5)
plt.plot(data_test, target_predicted, label="Linear regression",
         linestyle="--")
plt.scatter(data_test[::3], target_predicted[::3], label="Test predictions",
            color="tab:orange")
plt.legend()
_ = plt.title("Prediction function using a LinearRegression")


print("----- Decision tree for regression : 1 level -----")
# Decision trees are non-parametric models: they do not make assumptions
# about the way data is distributed

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=1)
tree.fit(data_train, target_train)
target_predicted = tree.predict(data_test)

plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, target_predicted, label="Decision tree")
plt.legend()
_ = plt.title("Prediction function using a DecisionTreeRegressor")

# The decision tree model does not have an a priori distribution for the data
# Our feature space was split into two partitions

from sklearn.tree import plot_tree

plt.figure()
_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(tree, feature_names=data_columns, ax=ax)

# The threshold for our feature (flipper length) is 206.5 mm.
# The predicted values on each side of the split are two constants: 3683.50 g
# and 5023.62 g. These values corresponds to the mean values
# of the training samples in each partition.

print("----- Decision tree for regression : 3 levels -----")

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(data_train, target_train)
target_predicted = tree.predict(data_test)

plt.figure()
sns.scatterplot(data=penguins, x="Flipper Length (mm)", y="Body Mass (g)",
                color="black", alpha=0.5)
plt.plot(data_test, target_predicted, label="Decision tree")
plt.legend()
_ = plt.title("Prediction function using a DecisionTreeRegressor")

from sklearn.tree import plot_tree

plt.figure()
_, ax = plt.subplots(figsize=(8, 6))
_ = plot_tree(tree, feature_names=data_columns, ax=ax)
