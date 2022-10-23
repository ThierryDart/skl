# -*- coding: utf-8 -*-
# MOOC FUN
# M11 - Tabular data exploration exo
# The goal with this data is to predict whether a person earns
# over 50K a year from heterogeneous data such as
# age, employment, education, family information, etc.

import pandas as pd

print("##### M11 - Tabular data exploration #####")


print("----- Loading the adult census dataset -----")


adult_census = pd.read_csv("./datasets/adult-census.csv")

print("Head:\n", adult_census.head())
# The column named class is our target variable.
# The two possible classes are <=50K (low-revenue) and >50K (high-revenue)

target_column = 'class'
print("Class:\n", adult_census[target_column].value_counts())

numerical_columns = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week']
categorical_columns = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country']
all_columns = numerical_columns + categorical_columns + [target_column]

adult_census = adult_census[all_columns]

print(f"The dataset contains {adult_census.shape[0]} samples and "
      f"{adult_census.shape[1]} columns")
print(f"The dataset contains {adult_census.shape[1] - 1} features.")

print("----- Visual inspection of the data -----")

print("--- Numerical variables")
_ = adult_census.hist(figsize=(20, 14))

print("--- Categorical variables")
print("Sex:\n", adult_census['sex'].value_counts())

print("Education:\n", adult_census['education'].value_counts())

# Croos tab to verify the same information
print("Education x education_num:\n",
      pd.crosstab(index=adult_census['education'],
                  columns=adult_census['education-num']))


import seaborn as sns
import matplotlib.pyplot as plt

print("--- Pairplot")
# We will plot a subset of the data to keep the plot readable
n_samples_to_plot = 5000
columns = ['age', 'education-num', 'hours-per-week']
_ = sns.pairplot(data=adult_census[:n_samples_to_plot], vars=columns,
                 hue=target_column, plot_kws={'alpha': 0.2},
                 height=3, diag_kind='hist', diag_kws={'bins': 30})
plt.show()


print("----- Creating decision rules by hand -----")

# Could create some hand-written rules that predicts
# whether someone has a high- or low-income.


_ = sns.scatterplot(
    x="age", y="hours-per-week", data=adult_census[:n_samples_to_plot],
    hue="class", alpha=0.5,
)
plt.show()

ax = sns.scatterplot(
    x="age", y="hours-per-week", data=adult_census[:n_samples_to_plot],
    hue="class", alpha=0.5,
)

age_limit = 27
plt.axvline(x=age_limit, ymin=0, ymax=1, color="black", linestyle="--")

hours_per_week_limit = 40
plt.axhline(
    y=hours_per_week_limit, xmin=0.18, xmax=1, color="black", linestyle="--"
)

plt.annotate("<=50K", (17, 25), rotation=90, fontsize=35)
plt.annotate("<=50K", (35, 20), fontsize=35)
_ = plt.annotate("???", (45, 60), fontsize=35)
plt.show()


