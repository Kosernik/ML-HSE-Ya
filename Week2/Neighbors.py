import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('../data/wine.data')

# print(data.head())

X = data.iloc[:, 1:]
y = data.iloc[:, 0]
# print(X.head())
# print(y.head())

model = KFold(shuffle=True, random_state=42, n_splits=5)

bestAccuracy = -1
bestNeighbors = 0

for i in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=i)
    cross = cross_val_score(classifier, X, y, cv=model, scoring='accuracy')
    mean = np.mean(cross)
    if mean > bestAccuracy:
        bestAccuracy = mean
        bestNeighbors = i
    # print(mean)

print("Best accuracy: ", bestAccuracy)
print("Best number of neighbors: ", bestNeighbors)

Xscaled = scale(X=X)

scaledBestAccuracy = -1
scalesBestNeigh = 0

for i in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=i)
    cross = cross_val_score(classifier, Xscaled, y, cv=model, scoring='accuracy')
    mean = np.mean(cross)
    if mean > scaledBestAccuracy:
        scaledBestAccuracy = mean
        scalesBestNeigh = i

print("Best scaled accuracy: ", scaledBestAccuracy)
print("Best scaled number of neighbors: ", scalesBestNeigh)

