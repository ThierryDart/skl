# -*- coding: utf-8 -*-
# MOOC FUN
# M11-exo - Tabular data exploration
# Predicting penguins species based on two of their body measurements:
# culmen length and culmen depth.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("##### M11-exo - Tabular data exploration #####")

print("----- Loading the dataset -----")
penguins = pd.read_csv("./datasets/penguins_classification.csv")


print("----- How many features are numerical? How many features are categorical -----")
print(penguins.head())

print("----- What are the different penguins species available in the dataset -----")
target_column = 'Species'
print("Class:\n", penguins[target_column].value_counts())


print("----- Plot histograms for the numerical features -----")
_ = penguins.hist(figsize=(8, 4))


print("----- How features distribution for each class -----")
pairplot_figure = sns.pairplot(penguins, hue="Species")
pairplot_figure = sns.pairplot(penguins, hue="Species", height=4)

