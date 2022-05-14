from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt

x = np.array([[-2,6], [-1,-4], [3,-1], [-3,-2], [-4,-5]])
y = np.array([1, 1, 1, 2, 3])

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x, y)

new_x = np.array([[-2, 0]])

prediction = knn.predict(new_x)

print(prediction)


plt.scatter(x[:,0], x[:,1], c=y)
plt.scatter(new_x[:,0], new_x[:,1], marker="x")
plt.grid()
plt.show()

