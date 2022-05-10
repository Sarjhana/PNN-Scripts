import numpy as np

# Add weights of each layer here
w = [
    # Layer 1 with four inputs going to three hidden nodes
    [[-0.7057, 1.9061, 2.6605, -1.1359], 
    [0.4900, 1.9324, -0.4269, -5.1570], 
    [0.9438, -5.4160, -0.3431, -0.2931]], 
    # Layer 2 with three inputs going to two output nodes
    [[-1.1444, 0.3115, -9.9812],
    [0.0106, 11.5477, 2.6479]]
    ]

# Biases for each output num for each layer
b = [[4.8432, 0.3973, 2.1761], [2.5230, 2.6463]]

# Inputs to the network
x = [[1,0,1,0],[0,1,0,1],[1,1,0,0]]

# Running through each layer of the network:

y_hat = []

# For each sample
for i in x:
    layer = i

    print("\nCalculating output for {}".format(i))

    # For each layer
    for j in range(len(w)):
        print("\nLayer {}".format(j))
        next_layer = np.zeros((len(w[j])))

        # For each set of weights in this layer
        for k in range(len(w[j])):
            print(w[j][k])
            next_layer[k] = np.dot(layer, w[j][k]) + b[j][k]
            print(next_layer)
        layer = next_layer

    y_hat.append(list(np.heaviside(layer, 0)))


output = list(zip(x, y_hat))

[print("Input: {}, output: {}".format(out[0], out[1])) for out in output]