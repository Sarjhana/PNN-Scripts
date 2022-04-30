# SEQUENTIAL WIDROW HOFF ALGO FOR WEEK 2 CODING EXCERCISE WITH QUESTION VALUES (NOT USING IRIS DATASET)

import numpy as np
# ------------------------------------------------------------------------------------
a = [1.0, 0.0, 0.0]           # Initial value of a/weights
b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Margin Vector
n = 0.1                 # Learning Rate
theta = 0.1               # Error
iterations = 12         # Iterations
AUTO_CONVERGE = False   # Ignore iterations varible, carry on until convergences

# Given Dataset:

# Add dataset as given in question. This is assuming that Sample Normalisation has NOT been applied:
X = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
Y = [1, 1, 1, -1, -1, -1]

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
# Sequential Widrow-Hoff Learning Algorithm
# Epoch for-loop: can directly put epoch numbers
for i in range(iterations):

    # This for-loop goes through each sample one-by-one:
    x, y, margin = list(zip(NORM_X, Y, b))[ i%len(list(zip(NORM_X, Y, b))) ]

    # Value of a to use. If first iteration, then uses parameters given in question:
    a_prev = a

    # Equation -> g(x) = a^{t}y
    ay = np.dot(a, x)

    a = a + (n * (margin - ay) * x)

    def converged():
        return sum((marginsum - np.dot(a, xsum)) * ysum for xsum, ysum, marginsum in zip(NORM_X, Y, b)) < theta

    if AUTO_CONVERGE and converged():
        break

print(a)
