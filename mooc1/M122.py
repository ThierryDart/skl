# -*- coding: utf-8 -*-
# MOOC FUN
# M122 - Working with numerical data
# 1 - identifying numerical data in a heterogeneous dataset;
# 2 - selecting the subset of columns corresponding to numerical data;
# 3 - using a scikit-learn helper to separate data into train-test sets;
# 4 - training and evaluating a more complex scikit-learn model.

import pandas as pd

print("##### M122 - Working with numerical data #####")

print("----- Loading the entire dataset -----")
adult_census = pd.read_csv("./datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")
print("Head:\n", adult_census.head())

# separates the target from the data
data, target = adult_census.drop(columns="class"), adult_census["class"]
print("Data:\n", data.head())
print("Target:\n", target.head())


print("----- Identify numerical data -----")
# Predictive models are natively designed to work with numerical data
# Numerical data are represented with numbers, but numbers are not always representing numerical data.
print("Data datatypes:\n", data.dtypes)
# We seem to have only two data types.
# We can make sure by checking the unique data types.
print("Data unique:", data.dtypes.unique())
# We see that the object data type corresponds to columns containing strings.
print("Data head:\n", data.head())

numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
print("Data numerical columns:\n", data[numerical_columns].head())
print("Data age describe:\n", data["age"].describe())

# store the subset of numerical columns in a new dataframe.
data_numeric = data[numerical_columns]


print("----- Train-test: split the dataset -----")
from sklearn.model_selection import train_test_split

# Automatically split the dataset into two subsets
# Random_state parameter allows to get deterministic results
# 25% of samples in the testing set, 75% in the training set
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25)

print(f"Number of samples in testing: {data_test.shape[0]} => "
      f"{data_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")

print(f"Number of samples in training: {data_train.shape[0]} => "
      f"{data_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")


# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(data_train, target_train)

accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")



