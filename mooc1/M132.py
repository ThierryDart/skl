# -*- coding: utf-8 -*-
# MOOC FUN
# M132 - Using numerical and categorical variables together
#

print("##### M132 - Using numerical and categorical variables together #####")

import pandas as pd

adult_census = pd.read_csv("./datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

# Split target and features
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])


print("----- Selection based on data types -----")
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

print("----- Dispatch columns to a specific processor -----")

# ColumnTransformer class will send specific columns to a specific transformer
# We first define the columns depending on their data type:
# - one-hot encoding will be applied to categorical columns with handle_unknown="ignore"
# - numerical scaling numerical features which will be standardized.

from sklearn.preprocessing import OneHotEncoder, StandardScaler
categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard-scaler', numerical_preprocessor, numerical_columns)])
print(preprocessor)

# A ColumnTransformer does the following:
# It splits the columns of the original dataset based on the column names or indices provided.
# It transforms each subsets. It will internally call fit_transform or transform
# It then concatenate the transformed datasets into a single dataset

# The important thing is that ColumnTransformer is like any other scikit-learn transformer.
# In particular it can be combined with a classifier in a Pipeline

print("----- Creating and using the pipeline -----")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

from sklearn import set_config
set_config(display='diagram')
print(model)

# Be aware that we use train_test_split here for didactic purposes
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

_ = model.fit(data_train, target_train)

print(data_test.head())
print(model.predict(data_test)[:5])
print(model.score(data_test, target_test))

print("----- Evaluation of the model with cross-validation -----")
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
print(cv_results)

scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")


print("----- Fitting a more powerful model -----")
# Linear models are nice because they are usually cheap to train, small to deploy, fast to predict and give a good baseline.

# More complex models such as an ensemble of decision trees can lead to higher predictive performance.
# like gradient-boosting trees, more precisely, HistGradientBoostingClassifier
# a large number of samples and limited number of informative features (e.g. less than 1000) with a mix of numerical and categorical variables.

# For tree-based models:
# - we do not need to scale the numerical features
# - using an ordinal encoding for the categorical variables is fine even if the encoding results in an arbitrary ordering

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)

preprocessor = ColumnTransformer([
    ('categorical', categorical_preprocessor, categorical_columns)],
    remainder="passthrough")

model = make_pipeline(preprocessor, HistGradientBoostingClassifier())

_ = model.fit(data_train, target_train)

print(model.score(data_test, target_test))


