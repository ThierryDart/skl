# -*- coding: utf-8 -*-
# MOOC FUN
# M14 Quizz
#

import pandas as pd

ames_housing = pd.read_csv("./datasets/house_prices.csv", na_values="?")
ames_housing = ames_housing.drop(columns="Id")

target_name = "SalePrice"
data, target = ames_housing.drop(columns=target_name), ames_housing[target_name]
target = (target > 200_000).astype(int)

print("Data info:\n", data.info())
print("Data head:\n", data.head())

print("", data[""].describe())

