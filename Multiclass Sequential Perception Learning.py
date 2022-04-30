# MULTICLASS SEQUENTIAL PERCEPTION LEARNING ALGO FOR WEEK 2 CODING EXCERCISE WITH QUESTION VALUES (NOT USING IRIS DATASET)

import numpy as np
# ------------------------------------------------------------------------------------
a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
             )         # Initial value of a/weights
n = 1                # Learning Rate
iterations = 2         # Iterations

# Given Dataset:

# Add dataset as given in question.
X = [[1, 1], [2, 0], [0, 2], [-1, 1], [-1, -1]]
Y = np.array([0, 0, 1, 1, 2])

# ------------------------------------------------------------------------------------
# Applying Sample Normalisation:
NORM_X = []

for x, y in zip(X, Y):
    x.insert(0, 1)  # Augmentation
    NORM_X.append(x)
NORM_X = np.asarray(NORM_X)

# ------------------------------------------------------------------------------------
# Multiclass Sequential Perceptron Learning Algorithm
# Epoch for-loop: can directly put epoch numbers
for o in range(iterations):

    # This for-loop goes through each sample one-by-one:
    x = NORM_X[o % len(NORM_X)]
    i = [o % len(NORM_X)]

    # Equation -> g(x) = a^{t}y
    def highestOutput(a, x):
        outputs = [np.dot(a_classes, x) for a_classes in a]
        index = np.argmax(outputs)
        maxcount = np.array([same == outputs[index]
                            for same in outputs]).sum()

        maxidx = index
        if maxcount > 0:
            for iddx, getmax in enumerate(outputs):
                if outputs[index] == getmax:
                    maxidx = iddx
        return maxidx

    idx = highestOutput(a, x)

    def isMisclassified(idx, Y):
        if idx != Y[i]:
            return True
        return False

    if isMisclassified(idx, Y):
        a[Y[i]] = a[Y[i]] + (n * x)
        a[idx] = a[idx] - (n * x)


print(a)
