# EG Week 2 Q6

import numpy as np 

a = [-25,6,3]

points = [[1,5],[2,5],[4,1],[5,1]]
labels = [1,1,2,2]

# Augmenting and normalising
y = [[1] + x if labels[points.index(x)] == 1 else list(np.dot([1] + x, -1)) for x in points]
y = np.array(y, dtype=np.float64)
print(y)

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