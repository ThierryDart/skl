# -*- coding: utf-8 -*-
# MOOC FUN
# M2

import pandas as pd

blood_transfusion = pd.read_csv("./datasets/blood_transfusion.csv")
target_name = "Class"
data = blood_transfusion.drop(columns=target_name)
target = blood_transfusion[target_name]

print("----- Q1 -----")
print("Head target:\n", target.head())
print("Uniquet target:\n", target.unique())
print("Count target:\n", target.value_counts(normalize=True))

print("----- Q2 -----")
# sklearn.dummy.DummyClassifier and the strategy "most_frequent"
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")

# Performing a 10-fold cross-validation
import pandas as pd
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dummy, data, target, cv=10)
print("Mean test error:\n", scores.mean())

print("----- Q3 -----")
# Repeat the previous experiment but compute the balanced accuracy
# instead of the accuracy score. Pass scoring="balanced_accuracy"
# when calling cross_validate or cross_val_score functions?
import pandas as pd
from sklearn.model_selection import cross_val_score

scores3 = cross_val_score(dummy, data, target, scoring="balanced_accuracy", cv=10)
print("Mean test error:\n", scores3.mean())

print("----- Q5 -----")
# Use a sklearn.neighbors.KNeighborsClassifier
# Create a scikit-learn pipeline where a StandardScaler, followed by a KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

pl = make_pipeline(StandardScaler(), KNeighborsClassifier())

print("k:", pl.get_params())

print("----- Q6 -----")
# Evaluate the previous model with a 10-fold cross-validation.
# Use the balanced accuracy as a score.
from sklearn.model_selection import cross_validate

cv_results = cross_validate(pl, data, target, cv=10,
            scoring="balanced_accuracy", return_train_score=True)
cv_results = pd.DataFrame(cv_results)
print(cv_results[["train_score", "test_score"]].mean())

print("----- Q7 -----")
# Study the effect of the parameter n_neighbors on the train and test score, using a validation curve
# Use a 5-fold cross-validation and compute the balanced accuracy score
# Plot the average train and test scores for the different value of the hyperparameter.

from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

param_range = [1, 2, 5, 10, 20, 50, 100, 200, 500]
param_name = "kneighborsclassifier__n_neighbors"
train_scores, test_scores = validation_curve(
    pl, data, target, param_name=param_name, param_range=param_range, cv=5,
    n_jobs=2, scoring="balanced_accuracy")

_, ax = plt.subplots()
for name, scores in zip(
    ["Training score", "Testing score"], [train_scores, test_scores]
):
    ax.plot(
        param_range, scores.mean(axis=1), linestyle="-.", label=name,
        alpha=0.8)
    ax.fill_between(
        param_range, scores.mean(axis=1) - scores.std(axis=1),
        scores.mean(axis=1) + scores.std(axis=1),
        alpha=0.5, label=f"std. dev. {name.lower()}")

ax.set_xticks(param_range)
ax.set_xscale("log")
ax.set_xlabel("Value of hyperparameter n_neighbors")
ax.set_ylabel("Balanced accuracy score")
ax.set_title("Validation curve of K-nearest neighbors")
ax.legend()
ax.plt()

