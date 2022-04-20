# Output negative signs are wrong - BUG

import pandas as pd
from sklearn.decomposition import PCA

def karhunen(data):
    
    # First we calculate the mean of each column
    means = np.zeros(len(data.columns))# for all columns
    for i in range(len(data.columns)):
        means[i] = data.iloc[:, i].mean()
    
    mean_zero = np.zeros(data.shape)
    # Now we substract the corresponding mean to all the values of our dataset
    for j in range(len(data)): # for all instances
        mean_zero[j, :] = data.iloc[j, :] - means
        
    # Now we have to calculate the covariance matrix
    cov = np.zeros((len(data.columns), len(data.columns)))
    for i in range(len(data)): # for all instances
        cov += np.array([mean_zero[i, :]]).T*mean_zero[i, :]
    cov = cov/len(data) # This is our final covariance matrix
    
    # Lastly, we obtain the eigenvalues and its corresponding eigenvectors
    eigenVal, eigenVec = np.linalg.eig(cov)
    
    return means, cov, eigenVal, eigenVec

# Modify Input Data

x = pd.DataFrame([[1., 2., 1.], [2, 3., 1.], [3., 5., 1.], [2., 2., 1.]]) # input data

means, cov, eigenVal, eigenVec = karhunen(x)

print("Means: {}\n\nCovariance Matrix: \n{}\n\nEigenValues: {}\n\nEigenVectors: \n{}".format(means, cov, eigenVal, eigenVec))

# We choose the two largest:
val = np.array([eigenVal[1], eigenVal[0]])
v = np.array([eigenVec[:, 1], eigenVec[:, 0]]).T
print("\n\nEigenValues for the 2 first principal components: {}\n\nEigenVectors for the 2 first principal components: \n{}".format(val, v))

# Now, for a point you use: 
y = np.dot(v.T, (x - means).T)
print("\n\nPoint before PCA: \n{}\n\nPoint with mean zero: \n{}\n\nProjection into the two first principal components: \n{}".format(np.array(x), np.array(x - means), y))

# FASTER WAY TO DO IT WITH SCIKIT LEARN
pca=PCA(n_components=2)
ytest=pca.fit(x).transform(np.array(x)) # for the first point
print("\n\nNew Points: \n{}".format(ytest.T))

# Now we have to calculate the covariance matrix
cov = np.zeros((y.shape[0],y.shape[0]))
for i in range(y.shape[1]): # for all instances
    cov += np.array([y[:, i]]).T*y[:, i]
cov = cov/y.shape[1] # This is our final covariance matrix
print("\nNew covariance matrix: \n{}".format(cov))
