# -*- coding: utf-8 -*-
# MOOC FUN
# M421-exo -
#
# Understand the parametrization of a linear model
# Quantify the fitting accuracy of a set of such models


print("##### M421-exo #####")


import pandas as pd

penguins = pd.read_csv("./datasets/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_name]], penguins[target_name]


print("-----3 linear models -----")

def linear_model_flipper_mass(
    flipper_length, weight_flipper_length, intercept_body_mass
):
    """Linear model of the form y = a * x + b"""
    body_mass = weight_flipper_length * flipper_length + intercept_body_mass
    return body_mass


import numpy as np
flipper_length_range = np.linspace(data.min(), data.max(), num=300)


import matplotlib.pyplot as plt
import seaborn as sns

weights = [-40, 45, 90]
intercepts = [15000, -5000, -14000]

ax = sns.scatterplot(data=penguins, x=feature_name, y=target_name,
                     color="black", alpha=0.5)

label = "{0:.2f} (g / mm) * flipper length + {1:.2f} (g)"
for weight, intercept in zip(weights, intercepts):
    predicted_body_mass = linear_model_flipper_mass(
        flipper_length_range, weight, intercept)

    ax.plot(flipper_length_range, predicted_body_mass,
            label=label.format(weight, intercept))

_ = ax.legend(loc='center left', bbox_to_anchor=(-0.25, 1.25), ncol=1)


print("----- Goodness of fit  -----")

def goodness_fit_measure(true_values, predictions):
    # we compute the error between the true values and the predictions of our
    # model
    errors = np.ravel(true_values) - np.ravel(predictions)
    # we can either square each error or take the absolute value:
    # these metrics are known as mean squared error (MSE)
    # and mean absolute error (MAE). Let's use the MAE here.
    return np.mean(np.abs(errors))

for model_idx, (weight, intercept) in enumerate(zip(weights, intercepts)):
    target_predicted = linear_model_flipper_mass(data, weight, intercept)
    print(f"Model #{model_idx}:")
    print(f"{weight:.2f} (g / mm) * flipper length + {intercept:.2f} (g)")
    print(f"Error: {goodness_fit_measure(target, target_predicted):.3f}\n")

