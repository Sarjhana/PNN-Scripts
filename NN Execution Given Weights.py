import numpy as np

# Add weights of each layer here
w = [
    # Layer 1 with four inputs going to three hidden nodes
    [[-0.62, 0.44, -0.91],[-0.81, -0.09, 0.02],[0.74, -0.91, -0.60],[-0.82, -0.92, 0.71],[-0.26, 0.68, 0.15],[0.80, -0.94, -0.83]], 
    # Layer 2 with three inputs going to two output nodes
    [[0.,0.,0.,-1.,0.,0.,2.]]
    ]

# Biases for each output num for each layer
# b = [[-0.62, -0.81, 0.74, -0.82, -0.26, 0.8], [0.]]

def augment(input):
    # If using augmented notation
    return [np.append([1], x) for x in input]


def linear(x, y):
    return x

activationFunctions = [np.heaviside, linear]

# Inputs to the network
x = np.array([[0,0],[0,1,],[1,0],[1,1]])
x = augment(x)
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

        # For each set of weights in this layer
        for k in range(len(w[j])):
            print(input, w[j][k])
            next_layer[k] = np.dot(input, w[j][k])

            # Use if not using augmented notation
            # next_layer[k] = np.dot(layer, w[j][k]) + b[j][k]
            print(next_layer)
        
        # If using augmentation
        if j < len(w)-1:
            next_layer = np.append([1], next_layer)
        
        input = activationFunctions[j](next_layer, 0)
        print(input, '\n')

    # Final layer, add any activation functions to z here.
    z.append(list(input))


output = list(zip(x, z))

[print("Input: {}, output: {}".format(out[0], out[1])) for out in output]