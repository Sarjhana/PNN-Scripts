# Extreme Learning Machine
# Tutorial example (Week 7 exercise 6)

# INPUTS:
    # x: input vectors (columns are instances and rows are features) WITHOUT AUGMENTATION
    # V: weights to hidden vectors
    # w: weights to the output neuron
    # activation_function: "linear_threshold" (only one done in tutorials)
    # augmentation: True/False
    
import numpy as np
    
def extreme_learning_machine(x, V, w, activation_function, augmentation):
    
    if augmentation:
        x = np.insert(x, 0, 1, axis=0)
        
    # We calculate VX
    response = np.dot(V, x)
    
    # And Y = H(VX)
    if activation_function == "linear_threshold":
        output = np.where(response>=0, 1, 0)
    
    if augmentation:
        output_augmented = np.insert(output, 0, 1, axis=0)
        
    # z = wY
    z = np.dot(w, output_augmented)
    
    return response, output, z
  
  
x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).T # input data
V = np.array([[-0.62, 0.44, -0.91], [-0.81, -0.09, 0.02], [0.74, -0.91, -0.6], [-0.82, -0.92, 0.71], [-0.26, 0.68, 0.15], [0.8, -0.94, -0.83]])
w = np.array([0, 0, 0, -1, 0, 0, 2])

response, output, z = extreme_learning_machine(x, V, w, "linear_threshold", True)
print("Response: \n{}\n\nOutput of Hidden Layer: \n{}\n\nFinal Output: \n{}".format(response, output, z))
