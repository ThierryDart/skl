# -*- coding: utf-8 -*-
# MOOC FUN
# M11 - Tabular data exploration
# The column "SalePrice" contains the target variable.
# A classification target to predict whether or not an house is expensive.
# "Expensive" is defined as a sale price greater than $200,000.


import pandas as pd
ames_housing = pd.read_csv("./datasets/house_prices.csv", na_values="?")
ames_housing = ames_housing.drop(columns="Id")

# Split features and target
target_name = "SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]
target = (target > 200_000).astype(int)

print("----- Q1: Dataset -----")
print(data.info())
print(data.head())
print(data.dtypes)

print("----- Q2, Q3: Dataset -----")
#Q2
print(len(data.columns))

#Q3
data_numbers = data.select_dtypes(["integer", "float"])
print(data_numbers.info())
print(len(data_numbers.columns))

print("----- Q5 : Pipeline and Model evaluation using cross-validation ----")

# Only numerical data
numerical_features = [
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
  "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
  "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
  "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

data_numerical = data[numerical_features]

model = make_pipeline(StandardScaler(), SimpleImputer(), LogisticRegression())
cv_results_num = cross_validate(model, data_numerical, target)
print(cv_results_num["test_score"].mean())

print("----- Q6 : Pipeline that can process both the numerical and categorical features  ----")

# numerical features should be processed as previously;
# The left-out columns should be treated as categorical variables using a sklearn.preprocessing.OneHotEncoder;
# Prior to one-hot encoding, insert the sklearn.impute.SimpleImputer(strategy="most_frequent") transformer to replace missing values by the most-frequent value in each column.

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = data.columns.difference(numerical_features)

categorical_processor = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"),
)
numerical_processor = make_pipeline(StandardScaler(), SimpleImputer())

preprocessor = make_column_transformer(
    (categorical_processor, categorical_features),
    (numerical_processor, numerical_features),
)
model = make_pipeline(preprocessor, LogisticRegression(max_iter=1000))
cv_results_all = cross_validate(model, data, target, error_score="raise")
print(cv_results_all["test_score"].mean())


print("Diff:", cv_results_all["test_score"].mean() - cv_results_num["test_score"].mean())

print("STD:", 3 * cv_results_all["test_score"].std())




