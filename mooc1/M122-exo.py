# -*- coding: utf-8 -*-
# MOOC FUN
# M122-exo Exercise M1.03
# To compare the statistical performance of our classifier (81% accuracy)
# to some baseline classifiers

import pandas as pd

print("##### M122-exo - Exercise M1.03 #####")


print("----- Loading the entire dataset -----")
adult_census = pd.read_csv("./datasets/adult-census.csv")


print("----- Split our dataset -----")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

print("----- Selecting only the numerical columns -----")
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]

print("----- Split the data and target into a train and test set -----")
from sklearn.model_selection import train_test_split
data_numeric_train, data_numeric_test, target_train, target_test = \
    train_test_split(data_numeric, target, random_state=42)

print(f"Number of samples in testing: {data_numeric_test.shape[0]} => "
      f"{data_numeric_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")

print(f"Number of samples in training: {data_numeric_train.shape[0]} => "
      f"{data_numeric_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")


from sklearn.dummy import DummyClassifier

class_to_predict = " >50K"
high_revenue_clf = DummyClassifier(strategy="constant", constant=class_to_predict)
high_revenue_clf.fit(data_numeric_train, target_train)
score = high_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only high revenue: {score:.3f}")


class_to_predict = " <=50K"
low_revenue_clf = DummyClassifier(strategy="constant", constant=class_to_predict)
low_revenue_clf.fit(data_numeric_train, target_train)
score = low_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting only low revenue: {score:.3f}")


print(adult_census["class"].value_counts())

print((target == " <=50K").mean())

most_freq_revenue_clf = DummyClassifier(strategy="most_frequent")
most_freq_revenue_clf.fit(data_numeric_train, target_train)
score = most_freq_revenue_clf.score(data_numeric_test, target_test)
print(f"Accuracy of a model predicting the most frequent class: {score:.3f}")


