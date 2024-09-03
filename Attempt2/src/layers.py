# layer.py
import numpy as np

class Layer:
    def __init__(self, num_nodes_in: int, num_nodes_out: int, activation: type):
        # Let num_nodes_in = m, num_nodes_out = n, i = number of inputs
        # Layers take a ixm matrix x and multiply it by weight matrix W of size mxn
        # bias vector b of size n is added to the new ixn matrix as an ixn matrix 
        #   where each column has the same values to get matrix z
        
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.activation = activation
        
        # Weight and biases
        self.W = np.random.randn(self.num_nodes_in, self.num_nodes_out)
        self.b = np.random.randn(self.num_nodes_out)
        
    def set_weights(self, weights: np.array) -> None:
        '''Sets weights'''
        if weights.shape != self.W.shape:
            raise Exception("Weight is incorrect shape")
        self.W = weights
        
    def set_biases(self, biases: np.array) -> None:
        '''Sets biases'''
        if biases.shape != self.b.shape:
            raise Exception("Bias is incorrect shape")
        self.b = biases
        
    def forward(self, X):
        '''Feeds input X forward through the layer'''
        # for a single input, z = Wa + b
        # Since we want it to be able to feed forward multiple inputs,
        #   z = aW^T + b where each column in the biases is the same and a is a matrix of inputs
        
        # let X = the activation of the previous layer and let a = the activation of the current layer
        self.X = X # ixm
        self.z = np.matmul(X, self.W) + self.b # ixn
        self.a = self.activation.function(self.z) # ixn
        return self.a # ixn
    
    def cost_function(self, X, y):
        '''Takes in an input (X) and its expected values (y) and returns the cost'''
        output = self.forward(X) # yhat, ixn
        # y(ixn) - output(ixn) = ixn
        # This new ixn matrix then has each term squared
        # Then its columns are summed to get an n sized row vector
        # Then each term is halved (still n sized vector)
        c = 0.5 * sum((y - output) ** 2) # cost, (size n)
        return c
    
    def calculate_output_weight_gradient(self, X, y):
        output = self.forward
        # dCdW = dCdA * dAdZ * dZdW
        # derivative of cost with respect to activation
                