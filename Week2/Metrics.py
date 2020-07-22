from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn.datasets as dataSets
from sklearn.preprocessing import scale
import numpy
import pandas



X, y = dataSets.load_boston(return_X_y=True)

# print(X[:5])
# print(y[:5])

X = scale(X=X)
y = scale(X=y)

# print(X[:5])
# print(y[:5])

p_vals = numpy.linspace(1.0, 10.0, num=200)

model = KFold(shuffle=True, random_state=42, n_splits=5)

bestAccuracy = None
bestP = 0

for i in p_vals:
    classifier = KNeighborsRegressor(n_neighbors=5, weights='distance')
    cross = cross_val_score(classifier, X, y, cv=model, scoring='neg_mean_squared_error')
    mean = numpy.mean(cross)
    print(mean)
    if bestAccuracy is None:
        bestP = i
        bestAccuracy = mean
    if mean > bestAccuracy:
        bestP = i
        bestAccuracy = mean


print("Best accuracy: ", bestAccuracy)
print("Best p: ", bestP)

