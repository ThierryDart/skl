# -*- coding: utf-8 -*-
# MOOC FUN
# M421 - Linear regression without scikit-learn
# Introduce linear regression
# Using the flipper length of a penguin, we would like to infer its mass.

print("##### M421 - Linear regression without scikit-learn #####")


import pandas as pd
penguins = pd.read_csv("./datasets/penguins_regression.csv")
print("head", penguins.head())

import seaborn as sns
import matplotlib.pyplot as plt
feature_names = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_names]], penguins[target_name]

plt.figure()
ax = sns.scatterplot(data=penguins, x=feature_names, y=target_name,
                     color="black", alpha=0.5)
ax.set_title("Flipper length in function of the body mass")



def linear_model_flipper_mass(flipper_length, weight_flipper_length,
                              intercept_body_mass):
    """Linear model of the form y = a * x + b"""
    body_mass = weight_flipper_length * flipper_length + intercept_body_mass
    return body_mass


# Check the body mass values predicted for a range of flipper lengths.
import numpy as np

weight_flipper_length = 45
intercept_body_mass = -5000

flipper_length_range = np.linspace(data.min(), data.max(), num=300)
predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)


label = "{0:.2f} (g / mm) * flipper length + {1:.2f} (g)"

plt.Figure()
ax = sns.scatterplot(data=penguins, x=feature_names, y=target_name,
                     color="black", alpha=0.5)
ax.plot(flipper_length_range, predicted_body_mass)
_ = ax.set_title(label.format(weight_flipper_length, intercept_body_mass))


# If the coefficient is negative, it means that penguins
# with shorter flipper lengths have larger body masses
weight_flipper_length = -40
intercept_body_mass = 13000

predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

plt.Figure()
ax = sns.scatterplot(data=penguins, x=feature_names, y=target_name,
                     color="black", alpha=0.5)
ax.plot(flipper_length_range, predicted_body_mass)
_ = ax.set_title(label.format(weight_flipper_length, intercept_body_mass))




body_mass_180 = linear_model_flipper_mass(
    flipper_length=180, weight_flipper_length=40, intercept_body_mass=0)
body_mass_181 = linear_model_flipper_mass(
    flipper_length=181, weight_flipper_length=40, intercept_body_mass=0)

print(f"The body mass for a flipper length of 180 mm "
      f"is {body_mass_180} g and {body_mass_181} g "
      f"for a flipper length of 181 mm")


weight_flipper_length = 25
intercept_body_mass = 0

# redefined the flipper length to start at 0 to plot the intercept value
flipper_length_range = np.linspace(0, data.max(), num=300)
predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)


plt.Figure()
ax = sns.scatterplot(data=penguins, x=feature_names, y=target_name,
                     color="black", alpha=0.5)
ax.plot(flipper_length_range, predicted_body_mass)
_ = ax.set_title(label.format(weight_flipper_length, intercept_body_mass))

weight_flipper_length = 45
intercept_body_mass = -5000

predicted_body_mass = linear_model_flipper_mass(
    flipper_length_range, weight_flipper_length, intercept_body_mass)

ax = sns.scatterplot(data=penguins, x=feature_names, y=target_name,
                     color="black", alpha=0.5)
ax.plot(flipper_length_range, predicted_body_mass)
_ = ax.set_title(label.format(weight_flipper_length, intercept_body_mass))

