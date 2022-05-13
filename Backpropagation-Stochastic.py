import numpy as np
import math

def symsigmoid(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def logsigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def LeakyReLU(x):
    alpha = 0.1
    return np.maximum(alpha*x, x)


def Heaviside(x):
    threshold = 0.1  # set threshold if needed, else set to 0
    return np.heaviside(x-threshold, 0.5)


def TanH(x):
    return np.tanh(x)


# Forward Prop
# --------------------------------------------
# CHANGE INPUTS 

# value of x
input = np.array([0.1, 0.9]) 

# input to hidden layer weights
inputWeights = np.array([[0.5, 0], [0.3, -0.7]]) 

# input to hidden layer bias
inputBias = np.array([0.2, 0]) 

# hidden to output layer weights
firstLayerWeights = np.array([[0.8, 1.6]])

# hidden to output layer bias
firstLayerBias = np.array([-0.4])

# change activation function if needed below
# -----------------------------------------------

# Net value of w*x+w0 = y for input to hidden layer
firstLayer_net = np.dot(inputWeights, input) + inputBias
print('Net input to hidden layer :', firstLayer_net)

# Activation for hidden layer --> f(firstLayer_net) 
firstLayerOutput = symsigmoid(firstLayer_net)

# Net value of hidden to output layer w*y+w0 = z
outputLayer_net = np.dot(firstLayerWeights, firstLayerOutput) + firstLayerBias

# Activation for output layer --> f(outputLayer_net)
outputLayer = symsigmoid(outputLayer_net)
print("Value of Z:", outputLayer)


# Backward Prop calculates for individual weights, if we need for entire matrix, replace numbers accordingly for each node
# --------------------------------------------
# CHANGE INPUTS 

# Output to Hidden only (reverse because backprop)
weight_before = -0.4 #weight of the node we want to update eg., m10 = -0.4
learning_rate = 0.25
output = -0.7869405   #Value of outputLayer
target = 0.5
hidden_output = 1
# --------------------------------------------

def derivative_output(x):
    return (4*math.exp(-2*x) / (1 + math.exp(-2*x))**2) #differential for tan sigmoid/symmetric sigmoid

weight = weight_before + \
    (-learning_rate * (output - target) * derivative_output(outputLayer_net) * hidden_output)
print("Updated weight for {} is {}". format(weight_before, np.round(weight, 4)))

# --------------------------------------------
# Output to Hidden to Input (reverse because backprop)
weight_before = 0.3
learning_rate = 0.25
output = -0.7869405 
target = 0.5 
input = 0.1
hidden_weight = 1.6 
net_ih = firstLayer_net[1] #SPECIFY 0 OR 1 DEPENDING ON WHICH HIDDEN NODE WE USE HERE
# --------------------------------------------

def derivative_hidden(x):
    return (4*math.exp(-2*x) / (1 + math.exp(-2*x))**2) #differential for tan sigmoid/symmetric sigmoid

# Sum of (target - output) * derivative_output(2) * hidden_weight
weight = weight_before + \
    (-learning_rate *  (output - target) * derivative_output(outputLayer_net) * hidden_weight * derivative_hidden(net_ih) * input)

print("Updated weight for {} is {}". format(weight_before, np.round(weight, 4)))
