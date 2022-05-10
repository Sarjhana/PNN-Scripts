import numpy as np

x = [[0.7, -0.6], [0.3, -0.7], [-0.4, -1.0], [0.4, 0.9], [0.5, 0.5], [0.1, -0.5], [-0.5, -0.4], [1.0, -0.6]]
y = np.array([1,0,0,0,0,1,1,1])

x_norm = [[1] + x_ for x_ in x]

# Take the negative of theta!
w = [-0.3, 0.5, -0.7]


predictions = [np.heaviside(np.dot(x_, w), 0) for x_ in x_norm]

print(y, predictions)

# print("True positive:", np.sum(y * predictions))