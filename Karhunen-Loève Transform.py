# 3 dimensional input ---> projection onto first 2 components
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt


# def karhunen(data):

#     # First we calculate the mean of each column
#     means = np.zeros(len(data.columns))  # for all columns
#     for i in range(len(data.columns)):
#         means[i] = data.iloc[:, i].mean()

#     mean_zero = np.zeros(data.shape)
#     # Now we substract the corresponding mean to all the values of our dataset
#     for j in range(len(data)):  # for all instances
#         mean_zero[j, :] = data.iloc[j, :] - means

#     # Now we have to calculate the covariance matrix
#     cov = np.zeros((len(data.columns), len(data.columns)))
#     for i in range(len(data)):  # for all instances
#         cov += np.array([mean_zero[i, :]]).T*mean_zero[i, :]
#     cov = cov/len(data)  # This is our final covariance matrix

#     # Lastly, we obtain the eigenvalues and its corresponding eigenvectors
#     eigenVal, eigenVec = np.linalg.eigh(cov)

#     return means, cov, eigenVal, eigenVec

# # Modify Input Data [1., 2., 1.], [2, 3., 1.], [3., 5., 1.], [2., 2., 1.]
x = pd.DataFrame([[4,2,2], [0,-2,2], [2,4,2], [-2,0,2]])  # input data

# means, cov, eigenVal, eigenVec = karhunen(x)

# print("Means: {}\n\nCovariance Matrix: \n{}\n\nEigenValues: {}\n\nEigenVectors: \n{}".format(
#     means, cov, eigenVal, eigenVec))

# # We choose the two largest:
# val = np.array([eigenVal[2], eigenVal[1]]) #
# v = np.array([eigenVec[:, 2], eigenVec[:, 1]]).T #
# print("\n\nEigenValues for the 2 first principal components: {}\n\nEigenVectors for the 2 first principal components: \n{}".format(val, v))

# # Now, for a point you use:
# y = np.dot(v.T, (x - means).T)
# print("\n\nPoint before PCA: \n{}\n\nPoint with mean zero: \n{}\n\n Projection into the 2 first principal components: \n{}".format(
#     np.array(x), np.array(x - means), y))

# # Now we have to calculate the covariance matrix
# cov = np.zeros((y.shape[0], y.shape[0]))
# for i in range(y.shape[1]):  # for all instances
#     cov += np.array([y[:, i]]).T*y[:, i]
# cov = cov/y.shape[1]  # This is our final covariance matrix
# print("\nNew covariance matrix: \n{}".format(cov))

pca = PCA(n_components=2)
pca.fit(x)
y = pca.transform(x)
print("\n\nZero-mean: \n{}".format((x - pca.mean_).T))
print("\n\nX projected: \n{}".format(y))
print("\n\nEigenvectors: \n{}".format(pca.components_))
print("\n\nEigenvalues: \n{}".format(pca.explained_variance_))
print("\n\nVariance explained: \n{}".format(pca.explained_variance_ratio_))
print("\n\nPCA: \n{}".format(pca.singular_values_))
print("\n\nMean: \n{}".format(pca.mean_))
print(pca.get_covariance())



x_ = np.array([[3, -2, 5]])
print("\nNew x: {}\nProjected: {}".format(x_, np.dot(pca.components_, x_[0])))


plt.scatter(y[:,0], y[:,1])
plt.show()


