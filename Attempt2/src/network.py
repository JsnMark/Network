# network.py

import numpy as np
import activations
import layers

class Network:
    def __init__(self, layer_sizes: list[int], activation_list: list[type]):
        if len(layer_sizes) != len(activation_list) + 1:
            raise Exception("length of layers must be one more than length of activations")
        
        self.layers = [layers.Layer(layer_sizes[i], layer_sizes[i+1], activation_list[i]) for i in range(len(layer_sizes) - 1)]
    
    def feed_forward(self, X):
        '''Takes input X and feeds it forward through the network to reach an output'''
        input = X
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def cost_function(self, X, y):
        '''Takes in an input (X) and its expected values (y) and returns the cost'''
        outputs = self.feed_forward(X) # yhat, ixn
        # y(ixn) - output(ixn) = ixn
        # This new ixn matrix then has each term squared
        # Then its columns are summed to get an n sized row vector
        # Then each term is halved (still n sized vector)
        c = 0.5 * sum((y - outputs) ** 2) # cost, (size n)
        return c
    
    def backpropagation(self, X, y):
        
        # dCdW = derivative of cost with respect to weight
        # dCdA = derivatve of cost with respect to activation
        # dAdZ = derivative of activation with respect to preactivation (z)
        # dZdW = derivative of z with respect to weight
        
        # dCdW = dCdW * dAdZ * dZdW
        
        # Calculate output layer gradients

        output_layer = self.layers[-1]
        
        dCdW = None