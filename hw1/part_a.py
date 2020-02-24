import numpy as np

A = np.array([[ 20,  20, 1], [ 20, 140, 1], [ 60,  60, 1]])
B = np.array([[230, 130, 1], [350, 190, 1], [290, 110, 1]])

T = np.linalg.solve(A, B).transpose()
print(T)