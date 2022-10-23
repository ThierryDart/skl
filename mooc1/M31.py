# -*- coding: utf-8 -*-
# MOOC FUN
# M31 - Set and get hyperparameters in scikit-learn
# Hyperparameters refer to the parameter that will control the learning process
#
# Should not be confused with the fitted parameters, resulting from the training,
# are recognizable because they are spelled with a final underscore _

print("##### M31 - Set and get hyperparameters in scikit-learn #####")

import pandas as pd

adult_census = pd.read_csv("./datasets/adult-census.csv")

target_name = "class"
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

target = adult_census[target_name]
data = adult_census[numerical_columns]

print("data.head:\n", data.head())

print("--- Create a simple predictive model made of a scaler followed by a logistic regression classifier")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", LogisticRegression())
])

print("--- Evaluate the statistical performance of the model via cross-validation")
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

print("--- Change the parameter")
# We can also change the parameter of a model after it has been created
# with the set_params method,
model.set_params(classifier__C=1e-3)
cv_results = cross_validate(model, data, target)
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} +/- {scores.std():.3f}")

# When the model of interest is a Pipeline, the parameter names are of the form
# <model_name>__<parameter_name> (note the double underscore in the middle)
# Use the get_params method on scikit-learn models to list all the parameters with their values.

for parameter in model.get_params():
    print(parameter)


# .get_params() returns a dict whose keys are the parameter names
# and whose values are the parameter values.
print("classifier__C:", model.get_params()['classifier__C'])

# Systematically vary the value of C to see if there is an optimal value.
for C in [1e-3, 1e-2, 1e-1, 1, 10]:
    model.set_params(classifier__C=C)
    cv_results = cross_validate(model, data, target)
    scores = cv_results["test_score"]
    print(f"Accuracy score via cross-validation with C={C}:\n"
          f"{scores.mean():.3f} +/- {scores.std():.3f}")