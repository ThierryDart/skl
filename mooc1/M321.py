# -*- coding: utf-8 -*-
# MOOC FUN
# M321 - Hyperparameter tuning by grid-search
# How to optimize hyperparameters using a grid-search approach

print("##### M321 - Hyperparameter tuning by grid-search #####")

print("----- Our predictive model -----")
from sklearn import set_config
set_config(display="diagram")

import pandas as pd
adult_census = pd.read_csv("./datasets/adult-census.csv")
# Extract the target
target_name = "class"
target = adult_census[target_name]
# Drop duplicate column
data = adult_census.drop(columns=[target_name, "education-num"])
print("data:\n", data.head())

# Split it into a training and testing sets
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)


# Define a pipeline as seen in the first module.
# It will handle both numerical and categorical features.

# Select all the categorical columns
from sklearn.compose import make_column_selector as selector
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

# Build our ordinal encoder
from sklearn.preprocessing import OrdinalEncoder
categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)

# Use a column transformer with code to select the categorical columns
# and apply to them the ordinal encoder.
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('cat-preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)

# Use a tree-based classifier (i.e. histogram gradient-boosting)
# to predict whether or not a person earns more than 50 k$ a year

# for the moment this line is required to import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
print(model)

print("----- Tuning using a grid-search -----")
# Instead of manually writing the loops, scikit-learn provides a class GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate': (0.05, 0.1, 0.5, 1, 5),
    'classifier__max_leaf_nodes': (3, 10, 30, 100)}
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train)

# check the accuracy of our model using the test set
accuracy = model_grid_search.score(data_test, target_test)
print(
    f"The test accuracy score of the grid-searched pipeline is: "
    f"{accuracy:.2f}"
)

# Be aware that the evaluation should normally be performed in a cross-validation
# framework by providing model_grid_search as a model to the cross_validate function.

# Once the grid-search is fitted, it can be used as any other predictor by calling
# predict and predict_proba. Internally, it will use the model with the best parameters found during fit.
print(model_grid_search.predict(data_test.iloc[0:5]))

# Can know about these parameters by looking at the best_params_ attribute
print(f"The best set of parameters is: "
      f"{model_grid_search.best_params_}")

# Inspect all results which are stored in the attribute cv_results_ of the grid-search.
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False)
print(cv_results.head())

# Focus on the most interesting columns and shorten the parameter names
# to remove the "param_classifier__" prefix for readability
# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += [
    "mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]

def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = cv_results.rename(shorten_param, axis=1)
print(cv_results)

# With only 2 parameters, we might want to visualize the grid-search as a heatmap.
# We need to transform our cv_results
pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score", index=["learning_rate"],
    columns=["max_leaf_nodes"])

print(pivoted_cv_results)

# Use a heatmap representation to show the above dataframe visually
import seaborn as sns

ax = sns.heatmap(pivoted_cv_results, annot=True, cmap="YlGnBu", vmin=0.7,
                 vmax=0.9)
ax.invert_yaxis()

