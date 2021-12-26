import numpy as np

a = np.array([i+1 for i in range(9)]).reshape((-1,3))
b = a[:,0]
print(b)
print(b.sum())