# -*- coding: utf-8 -*-
# MOOC FUN
# M131 - Encoding of categorical variables
# 1 - dealing with categorical variables by encoding them,
#     namely ordinal encoding and one-hot encoding.

import pandas as pd

print("##### M131 - Encoding of categorical variables #####")

adult_census = pd.read_csv("./datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

# Split target and features
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])


print("----- Identify categorical variables -----")
print("native-country:\n", data["native-country"].value_counts().sort_index())

print("Data types:\n", data.dtypes)


print("----- Select features based on their data type -----")

# use the function make_column_selector to select columns based on their data type
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
print("Categorical_columns:\n", categorical_columns)

# Filter out the unwanted columns:
data_categorical = data[categorical_columns]
print("Data_categorical:\n", data_categorical.head())

print(f"The dataset is composed of {data_categorical.shape[1]} features")

print("----- Strategies to encode categories -----")

print("--- Encoding ordinal categories")
# By default, OrdinalEncoder uses a lexicographical strategy
from sklearn.preprocessing import OrdinalEncoder

# encode education category with a different number
education_column = data_categorical[["education"]]
encoder = OrdinalEncoder()
education_encoded = encoder.fit_transform(education_column)
print("Education_encoded:\n", education_encoded)
# check the mapping
print(encoder.categories_)

#
data_encoded = encoder.fit_transform(data_categorical)
print("data_encoded:\n", data_encoded[:5])
print("encoder.categories_:\n", encoder.categories_)
print(f"The dataset encoded contains {data_encoded.shape[1]} features")

print("--- Encoding nominal categories (without assuming any order)")

# OneHotEncoder is an alternative encoder that prevents the downstream models
# it will create as many new columns as there are possible categories.

from sklearn.preprocessing import OneHotEncoder

# Education
# Sparse matrices are efficient data structures when most of your matrix elements are zero
encoder = OneHotEncoder(sparse=False)
education_encoded = encoder.fit_transform(education_column)
print("education_encoded:\n", education_encoded)

# Better understanding using the associated feature names resulting from the transformation.
feature_names = encoder.get_feature_names(input_features=["education"])
education_encoded = pd.DataFrame(education_encoded, columns=feature_names)
print("education_encoded:\n", education_encoded)
print(f"The dataset is composed of {data_categorical.shape[1]} features")

# apply this encoding on the full dataset
data_encoded = encoder.fit_transform(data_categorical)
print("data_encoded:\n", data_encoded[:5])
print(f"The encoded dataset contains {data_encoded.shape[1]} features")

# wrap this NumPy array in a dataframe with informative column names""
columns_encoded = encoder.get_feature_names(data_categorical.columns)
print("Data encoded:\n", pd.DataFrame(data_encoded, columns=columns_encoded).head())



print("--- Choosing an encoding strategy")

# In general OneHotEncoder is the encoding strategy used when the downstream models are linear models
# while OrdinalEncoder is used with tree-based models.

# You still can use an OrdinalEncoder with linear models but you need to be sure that:
# - the original categories (before encoding) have an ordering;
# - the encoded categories follow the same ordering than the original categories.

# Also, there is no need to use an OneHotEncoder even if the original categories do not have an given order with tree-based model.


print("----- Evaluate our predictive pipeline -----")
# let's train a linear classifier on the encoded data and check
# the statistical performance of this pipeline using cross-validation.

print("native-country:\n", data["native-country"].value_counts())

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
    )

# check the model's statistical performance only using the categorical columns.
from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data_categorical, target)
print("cv_results:\n", cv_results)

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} +/- {scores.std():.3f}")

