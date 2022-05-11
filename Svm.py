# SVM for two classes - binary classifier
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt


# REPLACE X AND Y ACCORDING TO THE QUESTION
X = np.array([[2,2], [2, -2], [-2,-2,], [-2,2], [1,1], [1,-1], [-1,-1], [-1,1]])
y = [1, 1, 1, 1, -1, -1, -1, -1]


def mapping(X):
    # Implement this funciton only if a mapping is required in the question.
    
    # phi = []
    # for x in X:
    #     if np.linalg.norm(x) > 2:
    #         # print(x, np.linalg.norm(x[0] - x[1]))
    #         x1 = 4 - (x[1]/2) + abs(x[0] - x[1])
    #         x2 = 4 - (x[0]/2) + abs(x[0] - x[1])
    #         phi.append([x1, x2])
    #     else:
    #         phi.append([x[0]-2, x[1]-3])

    # print(phi)

    # return np.array(phi)
    return X


X = mapping(X)

def _svm(X, y, support_vectors, support_vector_class):
    X = np.array(X)
    y = np.array(y)

    print("-"*100)
    
    w = []
    for idx in range(len(support_vectors)):
        w.append(support_vectors[idx] * support_vector_class[idx])
    w = np.array(w)
    eq_arr = []
    for idx, sv in enumerate(support_vectors):
        tmp = ((w @ sv) * support_vector_class[idx])
        tmp = np.append(tmp, [support_vector_class[idx]])
        eq_arr.append(tmp)
    eq_arr.append(np.append(support_vector_class, [0]))
    rhs_arr = [1] * len(support_vector_class)
    rhs_arr.extend([0])
    rhs_arr = np.array(rhs_arr)
    try:
        ans = rhs_arr @ np.linalg.inv(eq_arr)
    except:
        print("Unable to do inverse, taking pseudo inverse")
        ans = rhs_arr @ np.linalg.pinv(eq_arr)
    print("lambda and w_0 values are ", ans)
    final_weight = []
    for idx in range(w.shape[0]):
        final_weight.append(w[idx] * ans[idx])
    final_weight = np.array(final_weight)
    final_weight = np.sum(final_weight, axis=0)
    print("Weights: ")
    print(final_weight)
    print("Margin: ")
    print(2/np.linalg.norm(final_weight))
    print("-"*100)




# Fitting the model. Change the kernel type if the graph looks wrong.
clf = svm.SVC(kernel="poly", degree=2)
# clf = svm.SVC(kernel="linear")
clf.fit(X, y)



# Specify own support vecotors here, or leave commented to use SkLearn's.
# support_vectors = np.array([[3,1], [3,-1], [1,0]])
# support_vector_class = np.array([1, 1, -1])
support_vectors = clf.support_vectors_
support_vector_class = clf.predict(support_vectors)

print("Support vectors are: {}".format(support_vectors))

# Also running the data through _svm() since it gives us the lambda values.
# Therefore this function does not impact the graph but outputs useful data.
_svm(X, y, support_vectors=support_vectors,
     support_vector_class=support_vector_class)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
for i, x in enumerate(support_vectors):
    ax.annotate(x, (x[0], x[1]))

plt.show()

print("Beware of floating point errors...")

