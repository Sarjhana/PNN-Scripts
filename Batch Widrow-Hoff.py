# BATCH WIDROW HOFF ALGO FOR WEEK 2 CODING EXCERCISE WITH QUESTION VALUES (NOT USING IRIS DATASET)

import numpy as np
# ------------------------------------------------------------------------------------
a = [-1.5, 5.0, -1]           # Initial value of a/weights
b = np.array([2.0, 2.0, 2.0, 2.0, 2.0])  # Margin Vector
n = 0.2                 # Learning Rate
theta = 0.1               # Error
iterations = 2         # Iterations
AUTO_CONVERGE = False   # Ignore iterations varible, carry on until convergences

# Given Dataset:

# Add dataset as given in question. This is assuming that Sample Normalisation has NOT been applied:
X = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
Y = np.array([1, 1, 1, -1, -1])

# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
NORM_X = []

for x, y in zip(X, Y):
    x.insert(0, 1)  # Augmentation
    if y == -1:
        x = [i * -1 for i in x]
    NORM_X.append(x)
NORM_X = np.asarray(NORM_X)

# ------------------------------------------------------------------------------------
# Batch Widrow-Hoff Learning Algorithm
# Epoch for-loop: can directly put epoch numbers
for o in range(iterations):

    a = a - (np.dot(NORM_X.T, np.dot(NORM_X, a) - b) * n)

    def converged():
        return sum((marginsum - np.dot(a, xsum)) * ysum for xsum, ysum, marginsum in zip(NORM_X, Y, b))

    if AUTO_CONVERGE and converged():
        break

print(a)
