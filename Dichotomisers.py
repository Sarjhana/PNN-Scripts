import numpy as np
import matplotlib.pyplot as plt


class Dichotomiser:

    def __init__(self, w):
        self.w = np.array(w)

        for v, i in zip(w, range(len(w)-1, 0, -1)):
            print("{}x^{} + ".format(v, i), end="")
        print(w[-1])
        print("\nCheck that the above matches the exam question", "\n")
    

    def y(self, x, verbose=False):
        total = []

        print("Input: {}".format(x)) if verbose else None

        # For each set of weights, so each term in the polynomial
        for index, vector in enumerate(self.w[:-1]):
            polynomial = len(self.w) - index -1

            # Finding x^polynomial
            poly_x = np.power(x, polynomial)
            print("x^{} = {}".format(polynomial, poly_x)) if verbose else None


            dot = np.dot(poly_x, vector)
            print("Dot of {} and {}: {}".format(poly_x, vector, dot), end="\n") if verbose else None

            x_hat = np.sum(dot)
            print("Sum: {}".format(x_hat), end="\n") if verbose else None
            total.append(float(x_hat))


        total.append(float(self.w[-1][0][0]))
        print("Total: ", end="") if verbose else None
        print(" + ".join(np.array(total, dtype=str)), end="") if verbose else None

        output = np.sum(total)

        print(" = {}".format(output)) if verbose else None

        print() if verbose else None

        return float(output)

    def hyperplane_predict(self, x):
        # TODO Generalise this for any degree hyperplane.
        
        if len(self.w) == 2:
            a = (-np.array(self.w[0][0])/self.w[0][1])*x 
            # print(a)
            return np.sum(a - (np.array(self.w[-1])/np.array(self.w[0][1])))
        # elif len(self.w) == 3:
        #     a = (-np.array(self.w[0])) * np.power(x, 2)
        #     print(a)
        #     return np.sum((-np.array(self.w[-1]) - a)/np.array(self.w[1]))
        else:
            return 0
        # return self.y(x)

    def predict_dataset(self, X, verbose=False):
        output = [self.y(x, verbose) for x in X]

        [print("Input: {}, output: {}".format(x, y)) for x, y in zip(X, output)]
        return output

    def plot(self, X, verbose=False):

        X = np.array(X)

        g_values = self.predict_dataset(X, verbose)

        x_1 = X[:,0]
        y_1 = X[:,1]

        minimum_x_for_graph = min(x_1) - (0.25*abs(min(x_1)))
        maximum_x_for_graph = max(x_1) + (0.25*abs(max(x_1)))

        x_range = np.arange(minimum_x_for_graph, maximum_x_for_graph, 0.1)

        hyperplane = [self.hyperplane_predict(x) for x in x_range]

        # print(list(zip(points, g_values)))

        fix, ax = plt.subplots()

        ax.plot(x_range, hyperplane)
        ax.scatter(x_1, y_1, c='r')

        for i, txt in enumerate(zip(X, g_values)):
            ax.annotate(str(txt), (x_1[i], y_1[i]))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # Enter data from the question here:

    points = np.array([[0,-1], [1,1]])

    # Polynomials decreasing in degree ->>>

    # EG w = (2,1), w0 = -5 turns into:
    # w = np.array([[[2],[1]], [[-5]]])

    # And A = [[2,1], [2,1]], b = [[1],[2], c = -3 turns into:
    w = [[[2, 5], [5, -8]], [[1], [2]], [[-3]]]


    w = np.array(w, dtype=object)

    clf = Dichotomiser(w)

    clf.predict_dataset(points, verbose=True)

    clf.plot(points)

    print(clf.y([0.4,10]))
