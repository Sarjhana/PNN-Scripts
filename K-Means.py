import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from Dichotomisers import Dichotomiser

X = np.array([[-1,3], [1,4], [0,5], [4,-1], [3,0], [5,1]])

centroids = np.array([[-1,3],[5,1]])

kmeans = KMeans(n_clusters=len(centroids), init=centroids).fit(X)

computed_centroids = kmeans.cluster_centers_

lda = LinearDiscriminantAnalysis()
lda.fit(X, kmeans.labels_)

# print(lda.coef_, lda.intercept_)

w = np.array([[[8], [-8]], [[0]]], dtype=object)

dich = Dichotomiser(w)

print(dich.w)

dich.plot(X, verbose=True)

# print(w)

fig, ax = plt.subplots()

plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(computed_centroids[:,0], computed_centroids[:,1], marker='*', c='red')
print(computed_centroids)
for c in kmeans.cluster_centers_:
    ax.annotate(str(c), (c[0], c[1]))
plt.grid()
plt.show()