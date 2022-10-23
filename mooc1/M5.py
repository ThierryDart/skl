# -*- coding: utf-8 -*-
# MOOC FUN
# M5 - QCM
#

print("##### M5 - QCM #####")

import pandas as pd
ames_housing = pd.read_csv("./datasets/house_prices.csv", na_values="?")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]

# The column "SalePrice" contains the target variable


numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]

print("Head target:\n", target.head())
print("data.describe():\n", data.describe())

print("--- Question 1")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


modell = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("imputer", SimpleImputer()),
    ("regression", LinearRegression()),
])

cv_resl = cross_validate(modell, data_numerical, target, scoring="r2",
                            return_estimator=True, cv=10, n_jobs=2)

print("R2 lr:", cv_resl["test_score"].mean())

modelt = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("regression", DecisionTreeRegressor())
])

cv_rest = cross_validate(modelt, data_numerical, target, scoring="r2",
                            return_estimator=True, cv=10, n_jobs=2)
print("R2 dt:", cv_rest["test_score"].mean())


print("--- Question 2")


import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


def plot_regression(model, X, y, ax=None):
    model.fit(X, y)

    X_test = pd.DataFrame(
        np.arange(X.iloc[:, 0].min(), X.iloc[:, 0].max()),
        columns=X.columns,
    )
    y_pred = model.predict(X_test)

    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(x=X.iloc[:, 0], y=y, color="black", alpha=0.5, ax=ax)
    ax.plot(X_test, y_pred, linewidth=4)

    return ax


print(modelt.get_params().keys())

param_grid = {"regression__max_depth": np.arange(1, 15, 1)}
gscvt = GridSearchCV(modelt, param_grid=param_grid)

plot_regression(gscvt, data_numerical, target)
_ = plt.title(f"Optimal depth found via CV: "
              f"{gscvt.best_params_['max_depth']}")



print("--- Question 3")
