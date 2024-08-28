# layer.py
import numpy as np
from random import random
from copy import deepcopy

# IMPORTANT
# 2-D numpy arrays are normal matrices of mxn
# 1-D numpy arrays are treated as column vectors (mx1 matrix) even though they look horizontal

class Layer:
    def __init__(self, num_nodes_in: int, num_nodes_out: int, activation: type):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.activation_function = activation()

        self.z_n = None # z(n)
        self.a_n = None # a(n)
        self.a_m = None # a(n-1)
        

        # Let m = num_nodes_in (number of outputs), n = num_nodes_out (number of inputs)
        # Each layer will have a total of m * n weights.
        # Let w_1_2 be the weight at the 2nd row and the 3rd col 
        # Weight matrix in math (size mxn, outputxinput):
        # ( w_0_0  w_0_1  ...  w_0_n  )
        # ( w_1_0  w_1_1  ...  w_1_n  )
        # (  ⋮       ⋮      ⋮     ⋮     )
        # ( w_m_0  w_m_1  ...  w_m_n  )     

        # initialize weight matrix with random weights
        # numpy documentation suggests to use np.array rather than np.matrix
        # numpy random to get random from standard deviation
        self.weights = np.array([[np.random.randn() for n in range(num_nodes_in)] for m in range(num_nodes_out)])
        self.weight_gradient_matrix = np.array([[0 for n in range(self.num_nodes_in)] for m in range(self.num_nodes_out)])
        
        # bias vector (mx1)
        self.biases = np.array([np.random.randn() for m in range(num_nodes_out)])
        self.bias_gradient = np.array([0 for m in range(num_nodes_out)])
        
    def feed_forward(self, input: np.array) -> np.array:
        '''Takes a single input vector in R^m and passes it through the layer to get a vector in R^n'''
        self.a_m = input # nx1
        # Let Act(x) be a function that takes in vector x and applies an activation function to each element in the vector
        # Let W be the weight matrix, a be the output vector after applying activations, b be the bias vector, i be the input vector
        # a = Act((W)(i) + b)
        
        @np.vectorize
        def activate(activation_func, output_value: float) -> float:
            '''Takes in a number and applies the activation function to it'''
            return activation_func(output_value)

        output = np.matmul(self.weights, input) + self.biases
        self.z_n = output # mx1
        activations = activate(self.activation_function.func, output)
        self.a_n = activations # mx1
        return activations
    
    def single_cost(self, actual_output: np.array, expected_output: np.array) -> float:
        '''Computes the cost of a single training example'''
        # assert(actual_output.shape == (self.num_nodes_out,)) # mx1
        # assert(expected_output.shape == (self.num_nodes_out,)) # mx1
        
        def square(n: float):
            return n * n
        
        # Let i be the input vector, o be the expected output vector, 
        # ^2 to denote applying the square function to each element in a vector,
        # and sum() to denote a function that takes all the elements in a vector and adds them all up
        # cost = sum( (i - o)^2 )
        #                                        mx1  ->  scalar
        return square(actual_output - expected_output).sum() 
    
    def single_cost_derivative(self, actual_output: np.array, expected_output: np.array) -> np.array:
        '''Derivative of the cost function with respect to activation'''
        # assert(actual_output.shape == (self.num_nodes_out,)) # mx1
        # assert(expected_output.shape == (self.num_nodes_out,)) # mx1
        return 2 * (actual_output - expected_output)
    
    def calculate_output_gradient(self, actual_output: np.array, expected_output: np.array) -> np.array:
        '''Calculates the gradients corresponding to weights and biases in output layer and returns 
        the dc_da_m (derivative of cost with respect to the previous layer)'''
        # w = weight, a = activation/output, z = activation before activation function
        # change in c with respect to w_n = change in z_n with respect to w_n (dz_n/dw_n)
        #                                 * change in a_n with respect to z_n (da_n/dz_n)
        #                                 * change in c with respect to a_n (dc/da_n)
        # dc/da_n = 2(a_[n-1] - y)
        # da_n/dz_n = activation_derivative(z_[n-1])
        # dz_n/dw_n = a_n
        
        @np.vectorize
        def activation_derivative(n):
            return self.activation_function.derivative(n)
        
        # inner product last two terms
        # (x1, x2, ..., xn) * (y1, y2, ..., yn)^T = (x1y1, x2y2, ..., xnyn)
        da_n__dz_n = activation_derivative(self.z_n)   # mx1 vector
        dc_da_n = self.single_cost_derivative(actual_output, expected_output) # mx1 vector
        dc_dz = da_n__dz_n * dc_da_n # mx1 vector
        
        # bias gradient = 1 * da/dz * dc/da
        self.bias_gradient = dc_dz # mx1
        
        # create matrix by multiplication 
        # (x1, x2, ..., xm)^T * (y1, y2, ..., yn) = [x1y1 x1y2 ... x1yn]
        #                                           [x2y1 x2y2 ... x2yn]
        #                                           [...  ...  ... ... ]
        #                                           [xmy1 xmy2 ... xmyn]
        
        np_dc_dz = dc_dz[:, np.newaxis] # numpy mx1 column vector (Previously, mx1 was represented as (m,) arrays not (m,1))
        np_a_m = self.a_m[np.newaxis, :] # numpy 1xn row vector
        
        dc_dw = np.matmul(np_dc_dz, np_a_m) # mx1 * 1xn == mxn matrix
        # Set weight gradient
        self.weight_gradient_matrix = dc_dw
        
        # return dc/da_m to be used instead of dc/da_n which is a simpler derivative of the cost funciton
        # Multiply each row of the matrix by each row(single value) of the vector and add the sums of each row
        #  mxn                    mx1      mxn                              nx1
        # [x1y1 x1y2 ... x1yn]   [z1 ]    [x1y1z1 x1y2z1 ... x1ynz1]       [x1y1z1 + x2y1z2 + ... + xmy1zm]
        # [x2y1 x2y2 ... x2yn]   [z2 ] =  [x2y1z2 x2y2z2 ... x2ynz2]   ->  [x1y2z1 + x2y2z2 + ... + xmy2zm]
        # [...  ...  ... ... ]   [...]    [...     ...   ... ...   ]       [...    + ...    + ... + ...   ]
        # [xmy1 xmy2 ... xmyn]   [zm ]    [xmy1zm xmy2zm ... xmynzm]       [x1ynz1 + x2ynz2 + ... + xmynzm]
        
        dc_da_m = np.sum(dc_dw * np_dc_dz, axis=0) # nx1
        return dc_da_m # nx1
    
    def calculate_hidden_gradient(self, dc_da_m):
        '''Calculates the gradients corresponding to weights and biases in hidden layer and returns 
        the dc_da_m (derivative of cost with respect to the previous layer)'''
        dc_da_n = dc_da_m # mx1 (it was nx1 in the context of the layer after this)
        
        @np.vectorize
        def activation_derivative(n):
            return self.activation_function.derivative(n)
        
        da_n__dz_n = activation_derivative(self.z_n)   # mx1 vector
        dc_dz = da_n__dz_n * dc_da_n # mx1 vector   
        
        # bias gradient = 1 * da/dz * dc/da
        self.bias_gradient = dc_dz # mx1
        
        np_dc_dz = dc_dz[:, np.newaxis] # numpy mx1 column vector (Previously, mx1 was represented as (m,) arrays not (m,1))
        np_a_m = self.a_m[np.newaxis, :] # numpy 1xn row vector
        
        dc_dw = np.matmul(np_dc_dz, np_a_m) # mx1 * 1xn == mxn matrix
        
        # weight gradient
        self.weight_gradient_matrix = dc_dw
    
        dc_da_m = np.sum(dc_dw * np_dc_dz, axis=0) # nx1
        return dc_da_m # nx1
    
    def update_using_gradient(self, learn_rate: float)-> None:
        '''Updates the weights and biases using the gradients'''
        self.weights -= learn_rate * self.weight_gradient_matrix 
        self.biases -= learn_rate * self.bias_gradient 