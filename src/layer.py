# layer.py
import numpy as np
from random import random


# IMPORTANT
# 2-D numpy arrays are normal matrices of mxn
# 1-D numpy arrays are treated as column vectors (mx1 matrix) even though they look horizontal

class Layer:
    def __init__(self, num_nodes_in: int, num_nodes_out: int, activation):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.activation = activation
        # Let m = num_nodes_in (number of inputs), n = num_nodes_out (number of neurons/outputs in layer)
        # Each layer will have a total of m * n weights.
        # Let w_1_2 be the weight at the 2nd row and the 3rd col 
        # Weight matrix in math (size mxn):
        # ( w_0_0  w_0_1  ...  w_0_n  )
        # ( w_1_0  w_1_1  ...  w_1_n  )
        # (  ⋮       ⋮      ⋮     ⋮     )
        # ( w_m_0  w_m_1  ...  w_m_n  )     


        # initialize weight matrix with random weights
        # numpy documentation suggests to use np.array rather than np.matrix
        self.weights = np.array([[random() for n in range(num_nodes_in)] for m in range(num_nodes_out)])
        
        # bias vector (mx1)
        self.biases = np.array([random() for m in range(num_nodes_out)])
        
        
    def feed_forward(self, input: np.array):
        '''Takes a vector of inputs in R^m and passes it through the layer to get a vector in R^n'''
        
        # Let Act(x) be a function that takes in vector x and applies an activation function to each element in the vector
        # Let W be the weight matrix, a be the output vector after applying activations, b be the bias vector, i be the input vector
        # a = Act( (W)(i) + b)
        
        # vecotrize makes it so this activation is applied to all elements in a numpy array
        @np.vectorize
        def activate_all(activation_func, output_value: float) -> float:
            '''Takes in a number and applies the activation function to it'''
            return activation_func(output_value)

        outputs = np.matmul(self.weights, input) + self.biases
        activations = activate_all(self.activation.func, outputs)
        return activations
        
