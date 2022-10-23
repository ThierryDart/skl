# -*- coding: utf-8 -*-
# MOOC FUN
# M3 - QCM
# How to optimize hyperparameters using a grid-search approach

print("##### M3 - QCM #####")

# Load dataset penguins
import pandas as pd

penguins = pd.read_csv("./datasets/penguins.csv")
columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"
# Remove lines with missing values for the columns of interestes
penguins_non_missing = penguins[columns + [target_name]].dropna()
data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]

print("----- Q1 -----")
print("Head target:\n", target.head())
print("Uniquet target:\n", target.unique())
print("Count target:\n", target.value_counts(normalize=True))

print("data.describe():\n", data.describe())


print("----- Q2 -----")
# Evaluate the pipeline using 10-fold cross-validation using the balanced-accuracy scoring metric to choose the correct statements.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

print("Pileline parameters:\n", model.get_params())


from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
print("M1", cv_results["test_score"], cv_results["test_score"].mean(), cv_results["test_score"].std())


model.set_params(classifier__n_neighbors=51)
cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
print("M2", cv_results["test_score"], cv_results["test_score"].mean(), cv_results["test_score"].std())


model.set_params(preprocessor=None, classifier__n_neighbors=5)
cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
print("M3", cv_results["test_score"], cv_results["test_score"].mean(), cv_results["test_score"].std())


print("----- Q3 -----")
# Use sklearn.model_selection.GridSearchCV to study the impact of the choice
# of the preprocessor and the number of neighbors on the 10-fold cross-validated balanced_accuracy metric.
# We want to study the n_neighbors in the range [5, 51, 101] and preprocessor in the range all_preprocessors.

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


all_preprocessors = [
    None,
    StandardScaler(),
    MinMaxScaler(),
    QuantileTransformer(n_quantiles=100),
    PowerTransformer(method="box-cox"),
]

# Split it into a training and testing sets
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# Cross validation

# Model
model = Pipeline(steps=[
    ("preprocessor", None),
    ("classifier", KNeighborsClassifier()),
])

print("Pileline parameters:\n", model.get_params())

from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessor': all_preprocessors,
    'classifier__n_neighbors': (5, 51, 101)
    }

model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=10, scoring="balanced_accuracy")
model_grid_search.fit(data_train, target_train)

# check the accuracy of our model using the test set
accuracy = model_grid_search.score(data_test, target_test)
print(
    f"The test accuracy score of the grid-searched pipeline is: "
    f"{accuracy:.2f}"
)
