# Oja's Learning Rule

# ---------------- CHANGE INPUTS BELOW ------------------------
x = np.array([[5,5,4.4,3.2], [6.2,7.,6.3,5.7],[5.5,5.0,5.2,3.2],[3.1,6.3,4.0,2.5],[6.2,5.6,2.3,6.1]])
w = np.array([[-0.2, -0.2, 0.2, -0.0]])
n = 0.01
iterations = 5 # 1 epoch = len(x) eg., 5
# -------------------------------------------------------------

x = x - np.mean(x,0)
for i in range(iterations):
  y = np.dot(w, x[i % len(x)].T)
  w = w + (n*y*(x[i%len(x)] - (y*w)))
print(w)
