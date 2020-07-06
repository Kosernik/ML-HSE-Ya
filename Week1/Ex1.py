import numpy as np

x = np.random.normal(loc=1, scale=10, size=(10, 5))
print(x)

means = np.mean(x, axis=0)
standarts = np.std(x, axis=0)
x_norm = (x - means) / standarts
print(x_norm)

sums = np.sum(x, axis=1)
print(sums)
print(np.nonzero(sums > 10))

A = np.eye(3)
B = np.eye(3)
print(A)
print(B)

AB = np.vstack((A, B))
BA = np.hstack((A, B))
print(AB)
print(BA)
