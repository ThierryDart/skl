# -*- coding: utf-8 -*-
# MOOC FUN
# M123 - Preprocessing for numerical features
# will still use only numerical features
# 1 - an example of preprocessing, namely scaling numerical variables;
# 2 - using a scikit-learn pipeline to chain preprocessing and model training;
# 3 - assessing the statistical performance of our model via cross-validation instead of a single train-test split.


import pandas as pd

print("##### M123 - Preprocessing for numerical features #####")

print("----- Data preparation -----")
adult_census = pd.read_csv("./datasets/adult-census.csv")

# to display nice model diagram
from sklearn import set_config
set_config(display='diagram')

# Split target and features
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

# Select only the numerical columns
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]

# Divide our dataset into a train and test sets
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)


print("----- Model fitting with preprocessing -----")
print(data_train.describe())

# Scaling (mean=0, standard dev=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data_train)

print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)
# scikit-learn convention: if an attribute is learned from the data, its name ends with an underscore (i.e. _),

# Perform data transformation
data_train_scaled = scaler.transform(data_train)
print("Data scaled:\n", data_train_scaled)

# Method fit_transform is a shorthand method to call successively fit and transform
data_train_scaled = scaler.fit_transform(data_train)
print("Data scaled:\n", data_train_scaled)

data_train_scaled = pd.DataFrame(data_train_scaled,
                                 columns=data_train.columns)
print("Data scaled describe:\n", data_train_scaled.describe())

# Combine these sequential operations with a scikit-learn Pipeline
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
print(model)
print(model.named_steps)

start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

predicted_target = model.predict(data_test)
print("Prediected target:", predicted_target[:5])

model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model[-1].n_iter_[0]} iterations")


#  Compare this predictive model with the predictive model which did not scale features.
model = LogisticRegression()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start

model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} "
      f"with a fitting time of {elapsed_time:.3f} seconds "
      f"in {model.n_iter_[0]} iterations")

print("----- Model evaluation using cross-validation -----")
# Cross-validation consists of repeating the procedure such that the training and testing sets are different each time.
# K-fold strategy: the entire dataset is split into K partitions

from sklearn.model_selection import cross_validate

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_result = cross_validate(model, data_numeric, target, cv=5)
print("Cross validate:\n", cv_result)

scores = cv_result["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
