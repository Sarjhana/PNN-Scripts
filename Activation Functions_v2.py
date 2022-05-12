import numpy as np

x = np.array([[1,0.5,0.2],[-1,-0.5,-0.2], [0.1, -0.1,0]])

def ReLU(x):
    return np.maximum(0, x)
  
def LeakyReLU(x):
    alpha = 0.1
    return np.maximum(alpha*x, x)

def Heaviside(x):
    threshold = 0.1 #set threshold if needed, else set to 0
    return np.heaviside(x-threshold, 0.5)
  
def TanH(x):
    return np.tanh(x)
  

print(ReLU(x))
