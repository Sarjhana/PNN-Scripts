# WG Week 2 Q1

import matplotlib.pyplot as plt
import numpy as np

w = [2,1]
w0 = -5

w = np.array(w, dtype=np.float64)


points = [[1,1], [2,2], [3,3]]

x_1 = [x[0] for x in points]
y_1 = [x[1] for x in points]

minimum_x_for_graph = min(x_1) - (0.25*abs(min(x_1)))
maximum_x_for_graph = max(x_1) + (0.25*abs(max(x_1)))

x_range = np.arange(minimum_x_for_graph, maximum_x_for_graph, 0.1)

def y(w, x):
    return (-w[0]/w[1])*x - (w0/w[1])

hyperplane = [y(w, x) for x in x_range]


g_value = [np.sum(np.dot(w, x) + w0) for x in points]
print(g_value, points)

print(list(zip(points, g_value)))

fix, ax = plt.subplots()

ax.plot(x_range, hyperplane)
ax.scatter(x_1, y_1, c='r')

for i, txt in enumerate(zip(points, g_value)):
    ax.annotate(str(txt), (x_1[i], y_1[i]))
plt.show()