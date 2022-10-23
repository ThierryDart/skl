# -*- coding: utf-8 -*-
# MOOC FUN
# M121-exo - Exercise M1.02
#  scikit-learn API: .fit(X, y)/.predict(X)/.score(X, y)

import pandas as pd

print("##### M121-exo Exercise M1.02 - First model with scikit-learn #####")


print("----- Loading the adult census dataset -----")
# The goal with this data is to predict whether a person earns
# over 50K a year from heterogeneous data such as
# age, employment, education, family information, etc.

adult_census = pd.read_csv("./datasets/adult-census-numeric.csv")
print("Head:\n", adult_census.head())

print("----- Separate the data and the target -----")
target_name = "class"
target = adult_census[target_name]
print("Target:\n", target.head())

data = adult_census.drop(columns=[target_name, ])
print("Data:\n", data.head())
print("Columns:\n", data.columns)
print(f"The dataset contains {data.shape[0]} samples and "
      f"{data.shape[1]} features")

print("----- Fit a model and make predictions -----")

# to display nice model diagram
from sklearn import set_config
# set_config(display='diagram')

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=50)  # default=5

print("Fit:\n", model.fit(data, target))

print("- Let's use our model to make some predictions using the same dataset")
target_predicted = model.predict(data)
print("Predicted:\n", target_predicted[:10])

print("- We can compare these predictions to the actual data...")
print("Actual data:\n", target[:10])
print("Check if the predictions agree with the real targets:\n", target[:10] == target_predicted[:10])
print(f"Number of correct prediction: "
      f"{(target[:10] == target_predicted[:10]).sum()} / 10")
print("Average success rate:", (target == target_predicted).mean())

print("----- Train-test data split -----")
# evaluate the trained model on data that was not used to fit it

adult_census_test = pd.read_csv('./datasets/adult-census-numeric-test.csv')

# Separate features and target
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns=[target_name, ])

print(f"The testing dataset contains {data_test.shape[0]} samples and "
      f"{data_test.shape[1]} features")


accuracy = model.score(data_test, target_test)
model_name = model.__class__.__name__

print(f"The test accuracy using a {model_name} is "
      f"{accuracy:.3f}")









