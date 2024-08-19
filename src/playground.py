import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])


print(a)
print()

b = np.array([1,2, 3])

print(b)
print()
d = np.matmul(a, b)
print(d)

@np.vectorize
def square(x):
    return x - 1

print(square(d))