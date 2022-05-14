import numpy as np



# Add weights of each layer here
# w = np.array([
#     # Layer 1 with four inputs going to three hidden nodes
#     [[-0.7057, 1.9061, 2.6605, -1.1359],
#     [0.4900, 1.9324, -0.4269, -5.1570],
#     [0.9438, -5.4160, -0.3431, -0.2931],],
#     # Layer 2 with three inputs going to two output nodes
#     [[-1.1444, 0.3115, -9.9812],
#     [0.0106, 11.5477, 2.6479]]
#     ])

# 1 row of weights = for each output node in the next layer

w = np.array([
    [[1,0], [0.5,-3]],
    [[6,7]]
    ])

# Biases for each output num for each layer
# b = [[-0.62, -0.81, 0.74, -0.82, -0.26, 0.8], [0.]]

b = np.array([[0, -2], [-8]])

def augment(input):
    # If using augmented notation
    return [np.append([1], x) for x in input]


def linear(x, y):
    return x

def sigmoid(x, y):
    return 1. / (1. + np.exp(-x))

def tanh(x, y):
    return np.tanh(x)

def logSig(x,y):
    return np.log(sigmoid(x,y ))


# Final actiavtion function is applied to z to get y
activationFunctions = [linear, linear, linear]

# Inputs to the network
x = np.array([[1, 5]])
# x = augment(x)
print(x)
# Running through each layer of the network:

z = []

# For each sample
for input in x:

    print("Calculating output for {}".format(input))

    # For each layer
    for j in range(len(w)):
        print("\nLayer {}".format(j))
        next_layer = np.zeros((len(w[j])))


        for k in range(len(w[j])):

            w_ = np.array(w[j][k])

            print(input, w_, b[j][k])

            next_layer[k] += np.matmul(input, w_) + b[j][k]
            print(next_layer)

        
        input = activationFunctions[j](next_layer, 0)
        print(input, '\n')

    # Any activation functions for z?
    z.append(list(input))



output = list(zip(x, z))

[print("Input: {}, output: {}".format(out[0], out[1])) for out in output]