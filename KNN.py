from sklearn.neighbors import KNeighborsClassifier
import numpy as np

x = np.array([[0.15, 0.35], [0.15, 0.28], [
    0.12, 0.2], [0.1, 0.32], [0.06, 0.25]])
y = np.array([1, 2, 2, 3, 3])

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x, y)
print(knn.predict([[0.1, 0.25]]))
