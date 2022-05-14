# EG Week 2 Q6

import numpy as np 
from matplotlib import pyplot as plt

a = [1,2]

points = [[1,2],[2,1],[3,3],[6,5],[7,8]]
labels = [1,1,1,2,2]

# Augmenting and normalising
y = [[1] + x if labels[points.index(x)] == 1 else list(np.dot([1] + x, -1)) for x in points]
y = np.array(points, dtype=np.float64)
print(y)

print("a*y: {}".format(np.dot(y, a)))


plt.scatter(y[:,0], y[:,1], c=labels)
plt.grid()
plt.show()


lr = 1

print("Initial a:", a)



for i in range(5):

    misclassified = []
    for j in range(len(points)):
        if np.sum(np.dot(y[j], a)) < 1:
            #Uncomment for batch learning
            # misclassified.append(list(y[j]))
            
            # Uncomment for sequential learning
            a = a + lr * y[j]

    # Uncomment for batch learning
    # a = a + lr * np.sum(misclassified, axis=0)
    print("Iteration:", i)
    print("Misclassified: {}".format(misclassified))
    print("a: {}\n".format(a))