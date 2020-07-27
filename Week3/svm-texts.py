import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
grid = {'C': np.power(10.0, np.arange(-5, 6))}

random = 241
X = newsgroups.data
y = newsgroups.target
# print(X)
# print(y)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

feature_mapping = vectorizer.get_feature_names()
# print(feature_mapping)
# print(len(feature_mapping))
# print(feature_mapping[12345])

cv = KFold(shuffle=True, n_splits=5, random_state=random)

clf = SVC(random_state=random, kernel='linear')
gs = GridSearchCV(clf, grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs.fit(X, y)
bestC = gs.best_params_
# print(type(bestC))
# print(bestC.get('C'))


model = SVC(C=bestC.get('C'), random_state=random, kernel='linear')
model.fit(X, y)
df = pd.DataFrame(np.transpose(abs(model.coef_.toarray())),
                  index=np.asarray(vectorizer.get_feature_names()),
                  columns=["col1"])
# print(df.head(10))
df.sort_values(by=['col1'], inplace=True, ascending=False)
print("Sorted")
print(df.head(10))

lst = df.head(10).keys()
print(type(lst))
print(lst)
