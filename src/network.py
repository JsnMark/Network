# network.py
import layer
import numpy as np

class Network:
    def __init__(self, layer_sizes: list, activation_list: list):
        '''Length of activation_list should be one less than layer_sizes'''
        # Example for layers=[2,3,4], activation_list = [ReLu, sigmoid]
        #   There will be 3 layers. Layer 1 has 2 nodes, Layer 2 has 3 nodes, Layer 3 has 4 nodes
        #   The first layer (input layer) is not represented as a layer. It has 2 nodes of input.
        #   The second layer (part of the hidden layers) takes 2 inputs and gives 3 outputs. Relu activation
        #   The third layer (output layer) takes in 3 inputs and gives 4 outputs. sigmoid activation
    
        self.layers = [layer.Layer(layer_sizes[i], layer_sizes[i+1], activation_list[i]) for i in range(len(layer_sizes) - 1)]
       
    def feed_forward(self, inputs: list[float]):
        '''Takes in a list of inputs and feeds them into the network which returns the output''' 
        # convert input list to column vector
        input_array = np.array(inputs)
        
        # For each layer, feed forward 
        # (Apply linear transformations to the each vector and apply an actication function to the transformed vector)
        for layer in self.layers:
            input_array = layer.feed_forward(input_array)
            
        return input_array