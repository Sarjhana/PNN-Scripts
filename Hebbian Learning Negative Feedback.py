import numpy as np

w = np.array([[1, 1, 0], [1, 1, 1]])
x = np.array([1, 1, 0])
y = np.array([0, 0])
alpha = 0.5
iterations = 5

for i in range(iterations):
    e = x - np.dot(w.T, y)
    y = y + (alpha * np.dot(w, e))

print(y)
