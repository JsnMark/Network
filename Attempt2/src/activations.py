# activations.py
import numpy as np

# When considering one input at a time to get one output(vector z)
# Let W = weight matrix, a = input vector, b = bias vector
# z = Wa + b

# When considering multiple inputs at a time to get multiple outputs(matrix z)
# let W = weight matrix (transposed from the weight matrix above),
# A = matrix of inputs where each row is an input
# B = matrix of biases. Each column's entries are identical
# z = AW + B

class Activation:
    def function(z):
        pass
    def derivative(z):
        pass

class Sigmoid(Activation):
    def function(z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)
    
class DoNothing(Activation):
    def function(z):
        return z
    
    def derivative(z):
        return 0