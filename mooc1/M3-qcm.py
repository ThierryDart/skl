# -*- coding: utf-8 -*-
# MOOC FUN
# M3-qcm - M3 QCM
# How to optimize hyperparameters using a grid-search approach

print("##### M3-qcm - M3 QCM #####")

print("--- Q2")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

print("Param:\n", pipeline.get_params())

# param_grid =   # complete this line in your answer
param_grid = {'classifier__C': [0.1, 1, 10]}

model = GridSearchCV(
    pipeline,
    param_grid=param_grid
).fit(X, y)

print(model.best_params_)

print("--- Q4")

import numpy as np
import pandas as pd
import plotly.express as px
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name
cv_results = pd.read_csv("./figures/randomized_search_results.csv",
                         index_col=0)

fig = px.parallel_coordinates(
    cv_results.rename(shorten_param, axis=1).apply({
        "learning_rate": np.log10,
        "max_leaf_nodes": np.log2,
        "max_bins": np.log2,
        "min_samples_leaf": np.log10,
        "l2_regularization": np.log10,
        "mean_test_score": lambda x: x}),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()

