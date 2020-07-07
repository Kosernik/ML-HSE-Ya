import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r'C:\MyProjects\python\ML-HSE-Ya\data\titanic.csv')
# print(data.head(10))


# PassengerId,Survived,Name,SibSp,Parch,Ticket,Cabin,Embarked
data.drop('PassengerId', axis=1, inplace=True)
# data.drop('Survived', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('SibSp', axis=1, inplace=True)
data.drop('Parch', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Embarked', axis=1, inplace=True)

data.dropna(axis=0, inplace=True)
data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

print(data.head(10))

decTree = DecisionTreeClassifier(random_state=241)

decTree.fit(data[['Pclass', 'Sex', 'Age', 'Fare']], data['Survived'])

importances = decTree.feature_importances_
lst = [i for i in importances]
lst.sort(reverse=True)
print(type(importances))
print(importances)

print(lst[:2])
