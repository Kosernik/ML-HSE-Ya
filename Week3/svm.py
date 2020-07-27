import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('../data/svm-data.csv', header=None)
# print(data)
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

model = SVC(C=100000, random_state=241, kernel='linear')
model.fit(X=X, y=y)
print(model.support_)

for i in model.support_:
    print(i+1)
