# -*- coding: utf-8 -*-
# MOOC FUN
# M323 - Cross-validation and hyperparameter tuning
# In the previous notebooks, we saw two approaches to tune hyperparameters: grid-search and randomized-search.
# Show how to combine such hyperparameters search with a cross-validation.

print("##### M323 - Cross-validation and hyperparameter tuning #####")

print("----- Our predictive model -----")
from sklearn import set_config
set_config(display="diagram")

import pandas as pd
adult_census = pd.read_csv("./datasets/adult-census.csv")
# Extract target
target_name = "class"
target = adult_census[target_name]
# remove duplicate
data = adult_census.drop(columns=[target_name, "education-num"])
print("Data:\n", data.head())

# Create the same predictive pipeline as seen in the grid-search section.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)


# for the moment this line is required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
print(model)

print("----- Include a hyperparameter search within a cross-validation -----")

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.05, 0.1),
    'classifier__max_leaf_nodes': (30, 40)}
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)

cv_results = cross_validate(
    model_grid_search, data, target, cv=3, return_estimator=True)