import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([7, 8, 9])

print(np.matmul(A,x).shape)
