from collections import Counter

import pandas as pd
import numpy


data = pd.read_csv(r'C:\MyProjects\python\ML-HSE-Ya\data\titanic.csv', index_col='PassengerId')

# print(data.head(10))

print('Задание 1')
ex1 = data['Sex'].value_counts()
print(ex1)
print(ex1[0], ex1[1])

# ex1fileName = 'files/exercise1.txt'
# ex1file = open(ex1fileName, 'w')
# ex1file.write(str(ex1[0]))
# ex1file.write(" ")
# ex1file.write(str(ex1[1]))
# ex1file.close()

print()
print('Задание 2')
survived = data['Survived'].value_counts()
print(survived)
res2 = round((survived[1]*100.0/(survived[0]+survived[1])), 2)
print(res2)
# with open('files/exercise2.txt', 'w') as f2:
#     f2.write(str(res2))


print()
print('Задание 3')
first_class = data['Pclass'].value_counts()
print(first_class)
res3 = round((first_class[1] * 100.0) / (first_class[1]+first_class[2]+first_class[3]), 2)
print(res3)
# with open('files/exercise3.txt', 'w') as f3:
#     f3.write(str(res3))


print()
print('Задание 4')
age_mean = round(data['Age'].mean(), 2)
age_average = round(data['Age'].median(), 2)
print(age_mean)
print(age_average)
# with open('files/exercise4.txt', 'w') as f4:
#     f4.write(str(age_mean))
#     f4.write(str(" "))
#     f4.write(str(age_average))

print()
print('Задание 5')
pearson = round(data['SibSp'].corr(data['Parch']), 2)
print(pearson)
# with open('files/exercise5.txt', 'w') as f5:
#     f5.write(str(pearson))

print()
print('Задание 6')

females = data[data['Sex'] == 'female']['Name']
print(females.head(10))
print("Length", str(len(females)))
names = [n.split() for n in females]
print(names)
print(len(names))
print(names[0])


def firstName(arr):
    for i in range(len(arr)):
        if arr[i] in ['Miss.', 'Mrs.']:
            if i < len(arr)-1:
                return arr[i+1]


splitted = [firstName(n) for n in names]
print(len(splitted))
print(splitted[0], splitted[1], splitted[10])

counted = Counter(splitted)
currMax = 0
currBest = ""
print("Number of names:" + str(len(counted)))
print(counted)

for name, count in counted.items():
    if count > currMax:
        currMax = count
        currBest = name

print(currBest)
print(currMax)

# with open('files/exercise6.txt', 'w') as f6:
#     f6.write(str(currBest))
