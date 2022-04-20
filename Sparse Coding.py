import numpy as np
import math

def sparse(p, VT, x, _lambda):
    p = np.array(p)
    VT = np.array(VT)
    x = np.array(x)
    r_error = []
    for p in projections:
        val = x - VT @ p
        for i in val:
          r_error.append(math.sqrt((i[0]**2) + (i[1]**2)))
        #r_error.append(np.linalg.norm(val) + _lambda*np.count_nonzero(p))
    print("RECONSTRUCTION ERRORS: ")
    print(r_error)
    print(projections[np.argmin(r_error)], " for sparse coding")


# Replace according to the question
# projections are nothing but y
projections = [[1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0]]

VT = [[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
      [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]

x = [[-0.05, -0.95]]

sparse(p=projections, VT=VT, x=x, _lambda=1)
