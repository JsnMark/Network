import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([7, 8, 9])

y = np.array([1, 2, 3])
@np.vectorize
def square(n: float):
    return n * n

new = x - y
print(new)
new = square(new)
print(new.sum())