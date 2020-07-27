import pandas as pd


data = pd.read_csv('../data/data-logistic.csv', header=None)
# print(data.head())

y = data.iloc[:, 0]
X = data.iloc[:, 1:]
# print("X")
# print(X.head(5))
# print("y")
# print(y.head(5))
