from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x, y)
print(knn.predict([[7.1,3.8,6.7,2.5]]))
