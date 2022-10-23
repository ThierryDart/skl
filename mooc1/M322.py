# -*- coding: utf-8 -*-
# MOOC FUN
# M322 - Hyperparameter tuning by randomized-search
# Grid-search approach has limitations: It does not scale when the number of parameters to tune is increasing
# Another method to tune hyperparameters called randomized search.

print("##### M322 - Hyperparameter tuning by randomized-search #####")

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

# Split it into a training and testing sets
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

# Create the same predictive pipeline as seen in the grid-search section
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
    ("classifier", HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4)),
])

print(model)

print("----- Tuning using a randomized-search -----")
# Can randomly generate the parameter candidates.
# The RandomizedSearchCV class allows for such stochastic search.
# It is used similarly to the GridSearchCV but the sampling distributions need to be specified instead of the parameter values.


from scipy.stats import loguniform

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization': loguniform(1e-6, 1e3),
    'classifier__learning_rate': loguniform(0.001, 10),
    'classifier__max_leaf_nodes': loguniform_int(2, 256),
    'classifier__min_samples_leaf': loguniform_int(1, 100),
    'classifier__max_bins': loguniform_int(2, 255),
}

model_random_search = RandomizedSearchCV(
    model, param_distributions=param_distributions, n_iter=10,
    cv=5, verbose=1,
)
print(model_random_search.fit(data_train, target_train))

# Compute the accuracy
accuracy = model_random_search.score(data_test, target_test)
print(f"The test accuracy score of the best model is "
      f"{accuracy:.2f}")

from pprint import pprint

print("The best parameters are:")
pprint(model_random_search.best_params_)

# Inspect the results using the attributes cv_results as we did previously.
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

# get the parameter names
column_results = [
    f"param_{name}" for name in param_distributions.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]

cv_results = pd.DataFrame(model_random_search.cv_results_)
cv_results = cv_results[column_results].sort_values(
    "mean_test_score", ascending=False)
cv_results = cv_results.rename(shorten_param, axis=1)
print(cv_results)

#
#
#

# model_random_search = RandomizedSearchCV(
#     model, param_distributions=param_distributions, n_iter=200,
#     n_jobs=2, cv=5)
# model_random_search.fit(data_train, target_train)
# cv_results =  pd.DataFrame(model_random_search.cv_results_)
# cv_results.to_csv("../figures/randomized_search_results.csv")

cv_results = pd.read_csv("./figures/randomized_search_results.csv",
                         index_col=0)

(cv_results[column_results].rename(
    shorten_param, axis=1).sort_values("mean_test_score"))

import numpy as np
import plotly.express as px

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

