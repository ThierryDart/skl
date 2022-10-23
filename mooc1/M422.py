# -*- coding: utf-8 -*-
# MOOC FUN
# M422 - Linear regression with scikit-learn
# When doing machine learning, you are interested in selecting the model
# which will minimize the error on the data available the most.
#

print("##### M422 - Linear regression with scikit-learn #####")

import pandas as pd

penguins = pd.read_csv("./datasets/penguins_regression.csv")
feature_names = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_names]], penguins[target_name]

print("----- Linear regression fit -----")

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(data, target)

# The instance linear_regression will store the parameter values
# in the attributes coef_ and intercept_

weight_flipper_length = linear_regression.coef_[0]
print("a:", weight_flipper_length)

intercept_body_mass = linear_regression.intercept_
print("b:", intercept_body_mass)

import numpy as np

flipper_length_range = np.linspace(data.min(), data.max(), num=300)
predicted_body_mass = (
    weight_flipper_length * flipper_length_range + intercept_body_mass)


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=data[feature_names], y=target, color="black", alpha=0.5)
plt.plot(flipper_length_range, predicted_body_mass)
_ = plt.title("Model using LinearRegression from scikit-learn")



print("----- Goodness of fit of a model -----")
#  two metrics : (i) the mean squared error and (ii) the mean absolute error

from sklearn.metrics import mean_squared_error
inferred_body_mass = linear_regression.predict(data)
model_error = mean_squared_error(target, inferred_body_mass)
print(f"The mean squared error of the optimal model is {model_error:.2f}")

from sklearn.metrics import mean_absolute_error
model_error = mean_absolute_error(target, inferred_body_mass)
print(f"The mean absolute error of the optimal model is {model_error:.2f} g")

# A mean absolute error of 313 means that in average,
# our model make an error of +/- 313 grams
# when predicting the body mass of a penguin given its flipper length.

