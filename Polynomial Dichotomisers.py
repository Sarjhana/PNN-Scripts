import numpy as np
import matplotlib.pyplot as plt

# Polynomials decreasing in degree ->>>
w = [[[2,1],[1,4]], [[1],[2]], [[-3]]]


# Representing the given weights in polynomial form
for v, i in zip(w, range(len(w)-1, 0, -1)):
    print("{}x^{} + ".format(v, i), end="")
print(w[-1])
print("Check that the above matches the exam question", "\n")

points = [[0,-1], [1,1]]


def y(x, w):

    total = 0

    print("Input: {}".format(x))

    # For each set of weights, so each term in the polynomial
    for index, vector in enumerate(w[:-1]):
        polynomial = len(w) - index -1

        # Finding x^polynomial        
        poly_x = np.power(x, polynomial)
        print("x^{} = {}".format(polynomial, poly_x))

        dot = np.dot(poly_x, vector)
        print("Dot of {} and {}: {}".format(poly_x, vector, dot), end="\n")

        x_hat = np.sum(dot)
        print("Sum: {}".format(x_hat), end="\n")
        total += x_hat

    
    total += w[-1]

    print("Total output for {}: {}\n".format(x, float(total)))

    return float(total)

    # return np.sum(np.dot(w[:-1], x) + w[-1])

g_value = [y(x, w) for x in points]

[print("Input: {}, output: {}".format(x, y)) for x, y in list(zip(points, g_value))]