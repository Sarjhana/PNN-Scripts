# EG week 2 Q5

import numpy as np
import matplotlib.pyplot as plt

# Polynomials decreasing in degree ->>>
a = [-3,1,2,2,2,4]

x = [[0,-1,0,0,1], [1,1,1,1,1]]

# Augmented notation
y = [[1] + x for x in x]
print(y)

def g(x, w):
    return np.sum(np.dot(w, x))

g_value = [g(x, a) for x in y]

print(list(zip(x, g_value)))