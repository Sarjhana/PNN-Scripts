# SEQUENTIAL PERCEPTION LEARNING ALGO FOR WEEK 2 CODING EXCERCISE WITH QUESTION VALUES (NOT USING IRIS DATASET)

import numpy as np
# ------------------------------------------------------------------------------------
a = np.array([1.0, 0.0, 0.0])         # Initial value of a/weights
n = 1                # Learning Rate
iterations = 5         # Iterations
DO_SAMPLE_NORMALISATION = True  # Should this code perform sample normalisation

# Given Dataset:

# Add dataset as given in question.
X = [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]]
Y = np.array([1, 1, 1, -1, -1, -1])

# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
NORM_X = []

for x, y in zip(X, Y):
    x.insert(0, 1)  # Augmentation
    if y == -1 and DO_SAMPLE_NORMALISATION:
        x = [i * -1 for i in x]
    NORM_X.append(x)
NORM_X = np.asarray(NORM_X)

# ------------------------------------------------------------------------------------
# Sequential Perceptron Learning Algorithm
# Epoch for-loop: can directly put epoch numbers
for i in range(iterations):

    # This for-loop goes through each sample one-by-one:
    x, y = list(zip(NORM_X, Y))[i % len(list(zip(NORM_X, Y)))]

    # Value of a to use. If first iteration, then uses parameters given in question:
    a_prev = a

    # Equation -> g(x) = a^{t}y
    ay = np.dot(a, x)

    def isMisclassified(ay, y):
        if DO_SAMPLE_NORMALISATION and ay <= 0:
            return True
        elif DO_SAMPLE_NORMALISATION:
            return False

        if ay <= 0 and y == 1:
            return True
        elif ay > 0 and y == -1:
            return True
        return False

    if isMisclassified(ay, y):
        if DO_SAMPLE_NORMALISATION:
            a = a + (n * x)
        else:
            a = a + (n * y * x)

print(a)
