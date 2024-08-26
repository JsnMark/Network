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
       
    def calculate_output(self, inputs: list[float]):
        '''Takes in a list of inputs and feeds them into the network which returns the output''' 
        # convert input list to column vector
        input_array = np.array(inputs)
        
        # For each layer, feed forward 
        # (Apply linear transformations to the each vector and apply an actication function to the transformed vector)
        for layer in self.layers:
            input_array = layer.feed_forward(input_array)
            
        return input_array
    
    def cost_function(self, training_data: list[np.array], expected_outputs: list[np.array]):
        '''Finds the average cost of training data. training_data elements 
        must correlate with expected_outputs elements at the same indexes'''
        last_layer = self.layers[-1]
        cost = 0.0
        # manually find len of data since redundant to go through the data again using len()
        data_len = 0
        for data_point, expected_output in zip(training_data, expected_outputs):
            cost += last_layer.single_cost(data_point, expected_output)
            data_len += 1
            
        cost = cost / data_len
        return cost
    
    
    def backpropogation(self, training_datas: list[np.array], expected_outputs: list[np.array]):
        '''Performs backpropogation, adjusting the layers's weights and biases'''
        output_layer = self.layers[-1]
        
        for training_data, expected_output in zip(training_datas, expected_outputs):
            actual_output = output_layer.feed_forward(training_data)
            dc_da_m = output_layer.calculate_output_gradient(actual_output, expected_output)
            
            for layer in self.layers[:-1]:
                dc_da_m = layer.calculate_hidden_gradient(dc_da_m)
        
        
        
        