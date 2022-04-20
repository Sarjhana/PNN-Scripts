# Output first row should be multiplied by a -ve sign --> BUG in code


import numpy as np
from scipy.linalg import svd


def transform(ip, n_components, data_to_project=[]):
    ip = np.array(ip)
    N, D = ip.shape
    ip_mean = np.mean(ip, axis=0)
    ip_prime = ip - ip_mean
    ip_prime = ip_prime.T
    C = (ip_prime) @ (ip_prime.T)
    C = C / N
    V, D, VT = svd(C)
    p_input = VT @ (ip_prime)
    print("-"*100)
    print("Diagonal matrix [EIGEN VALUES]")
    print(np.round(np.diag(D), 3))
    print()
    print("EIGEN VECTOR[TRANSPOSED]")
    print(VT)
    print()
    print(
        "PROJECTION OF INPUT [READ  COLUMNS FROM TOP, TILL N_COMPONENTS ROWS]")
    print(np.round(p_input, 3))
    if len(data_to_project) > 0:
        print("PROJECTION OF TEST DATA")
        p_given_data = VT @ data_to_project.T
        print("READ COLUMN WISE")
        print(p_given_data)

# Modify Input Data
input = [[1, 2, 1], [2, 3, 1], [3, 5, 1], [2, 2, 1]]
data_to_project = []
n_components = 2
transform(input, n_components=n_components, data_to_project=data_to_project)
