# BATCH DELTA LEARNING ALGO FOR WEEK 2 CODING EXCERCISE WITH QUESTION VALUES (NOT USING IRIS DATASET)

import numpy as np
# ------------------------------------------------------------------------------------
a = np.array([-1.5, 5.0, -1.0])         # Initial value of a/weights
n = 1                # Learning Rate
epochs = 3         # Epochs
DO_SAMPLE_NORMALISATION = False

# Given Dataset:

# Add dataset as given in question.
X = [[0, 0], [1, 0], [2, 1], [0, 1], [1, 2]]
Y = np.array([1, 1, 1, 0, 0])

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
# Batch Delta Learning Algorithm
# Epoch for-loop: can directly put epoch numbers
for i in range(epochs):

    # This for-loop goes through each sample one-by-one:
    TOTAL_ERROR = np.zeros(len(a))
    for x, y in zip(NORM_X, Y):

        # Value of a to use. If first iteration, then uses parameters given in question:
        a_prev = a

        # Equation -> g(x) = a^{t}y
        ay = np.heaviside(np.dot(a, x), 0)

        def isMisclassified(ay, y):
            if ay != y:
                return True
            return False

        if isMisclassified(ay, y):
            TOTAL_ERROR += (n * (y - ay) * x)

    a = a + TOTAL_ERROR

print(a)
