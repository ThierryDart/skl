# -*- coding: utf-8 -*-
# MOOC FUN
# M131 - Exercise M1.04
# to evaluate the impact of using an arbitrary integer encoding for categorical variables
# along with a linear classification model such as Logistic Regression.
# OrdinalEncoder / OneHotEncoder for Logistic Regression

print("##### M131 - Exercise M1.04 #####")

import pandas as pd

adult_census = pd.read_csv("./datasets/adult-census.csv")

# Split target and features
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

# get only the columns containing strings (column with object dtype)
from sklearn.compose import make_column_selector as selector
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]


print("-----Pipeline 1 : OrdinalEncoder and LogisticRegression -----")
# Pipeline composed of an OrdinalEncoder and a LogisticRegression classifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

model1 = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    LogisticRegression(max_iter=500)
    )

# check the model's statistical performance
from sklearn.model_selection import cross_validate
p1_results = cross_validate(model1, data_categorical, target)
print("p1_results:\n", p1_results)

scores1 = p1_results["test_score"]
print(f"The accuracy is: {scores1.mean():.3f} +/- {scores1.std():.3f}")


print("-----Pipeline 1b : Dummy -----")
from sklearn.dummy import DummyClassifier

p1b_results = cross_validate(DummyClassifier(strategy="most_frequent"),
                            data_categorical, target)
scoresp1b = p1b_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scoresp1b.mean():.3f} +/- {scoresp1b.std():.3f}")




print("-----Pipeline 2 : OneHotEncoder and LogisticRegression -----")
from sklearn.preprocessing import OneHotEncoder

model2 = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
    )

# check the model's statistical performance only using the categorical columns.
from sklearn.model_selection import cross_validate
p2_results = cross_validate(model2, data_categorical, target)
print("p2_results:\n", p2_results)

scores2 = p2_results["test_score"]
print(f"The accuracy is: {scores2.mean():.3f} +/- {scores2.std():.3f}")

# ---> linear model and OrdinalEncoder are used together
# only for ordinal categorical features, features with a specific ordering.
# Otherwise, your model will perform poorly.
