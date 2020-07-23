import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


testData = pd.read_csv('../data/perceptron-test.csv', header=None)
trainData = pd.read_csv('../data/perceptron-train.csv', header=None)
random = 241

Xtest = testData.iloc[:, 1:]
Ytest = testData.iloc[:, 0]
# print(Xtest.head())
# print(Ytest.head())

Xtrain = trainData.iloc[:, 1:]
Ytrain = trainData.iloc[:, 0]
# print(Xtrain.head())
# print(Ytrain.head())

clf = Perceptron(random_state=random, max_iter=5, tol=None)
clfScaled = Perceptron(random_state=random, max_iter=5, tol=None)

clf.fit(Xtrain, Ytrain)
predictions = clf.predict(Xtest)
# print(predictions)

accuracyUnscaled = accuracy_score(Ytest, predictions)
print(accuracyUnscaled)

scaler = StandardScaler()
XtrainScaled = scaler.fit_transform(Xtrain)
XtestScaled = scaler.transform(Xtest)

clfScaled.fit(XtrainScaled, Ytrain)
predictionsScaled = clfScaled.predict(XtestScaled)
# print(predictionsScaled)

accuracyScaled = accuracy_score(Ytest, predictionsScaled)
print(accuracyScaled)

print("Answer is: ", (accuracyScaled-accuracyUnscaled))
