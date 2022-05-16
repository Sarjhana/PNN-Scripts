import numpy as np

# -------------------------------------------------------------
# ------------ Oja's Learning Rule - SEQUENTIAL ---------------
# ---------------- CHANGE INPUTS BELOW ------------------------
x = np.array([[0,1], [3,5], [5,4], [5,6], [8,7], [9,7]])
w = np.array([[-1,0]])
n = 0.01
iterations = 6 # 1 epoch = len(x) eg., 5
# -------------------------------------------------------------

x = x - np.mean(x,0)

# Updates for every iteration/input
for i in range(iterations):
  y = np.dot(w, x[i % len(x)].T)
  w = w + (n*y*(x[i%len(x)] - (y*w)))
print(w)

# -------------------------------------------------------------
# -------------- Oja's Learning Rule - BATCH ------------------
# -------------------------------------------------------------
# ---------------- CHANGE INPUTS BELOW ------------------------
x = np.array([[0,1], [3,5], [5,4], [5,6], [8,7], [9,7]])
w = np.array([[-1,0]])
n = 0.01
epochs = 6
# -------------------------------------------------------------

x = x - np.mean(x,0)

# Updates only once per epoch
for o in range(epochs):
  inter = 0
  for i in range(len(x)):
    y = np.dot(w, x[i].T)
    inter = inter + n*y*(x[i] - (y*w))
  w = w + inter

print(w)

'''
# If question asks for projecting zero-mean data onto first component
y = []
for i in x:
  y.append(np.dot(w,i))
print(y)'''
