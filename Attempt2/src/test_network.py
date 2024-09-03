# test_network.py

import unittest
import numpy as np
from activations import *
from layers import *
from network import *

class TestNetwork(unittest.TestCase):
    def test_network_has_correct_parameters(self):
        layers = [2,3,1]
        activations = [Sigmoid, Sigmoid]
        
        network = Network(layers, activations)
        self.assertEqual(type(network.layers), list)
        self.assertEqual(len(network.layers), 2)
        self.assertEqual(type(network.layers[0]), Layer)
        self.assertEqual(type(network.layers[1]), Layer)
        
    def test_network_raises_error_on_improper_sizes(self):
        layers = [2,3,1]
        activations = [Sigmoid]
        with self.assertRaises(Exception):
            Network(layers, activations)
            
        layers = [2,3,1]
        activations = [Sigmoid, Sigmoid, Sigmoid]
        with self.assertRaises(Exception):
            Network(layers, activations)
            
    def test_feed_forward(self):
        layers = [2,3,1]
        activations = [DoNothing, DoNothing]
        network = Network(layers, activations)
        
        inputs = np.array([[1,2],
                           [3,4],
                           [2,1],
                           [4,3]])
        
        network.layers[0].set_weights(np.array([[0,1,2],
                                               [3,4,5]]))
        network.layers[0].set_biases(np.array([3,2,1]))
        network.layers[1].set_weights(np.array([[0],
                                                [1],
                                                [0]]))
        network.layers[1].set_biases(np.array([-1]))
        
        output = network.feed_forward(inputs)
        expected_output = np.array([[10], [20], [7], [17]]) # calculated by

        self.assertTrue(np.array_equal(output, expected_output))
        
    def test_cost_function(self):
        inputs = np.array([[0,0],
                           [1,2],
                           [4,5],
                           [2,1]])
        weights = np.array([[0,2,4],
                            [1,3,5]])
        biases = np.array([0,1,2])
        
        network = Network([2,3], [DoNothing])
        network.layers[0].set_weights(weights)
        network.layers[0].set_biases(biases)
        
        expected_outputs = np.array([[0,0,1],
                                     [0,0,1],
                                     [1,0,0],
                                     [1,1,0]])
        expected_cost = np.array([10, 353.5, 1150])
        cost = network.cost_function(inputs, expected_outputs)    
        self.assertTrue(np.allclose(expected_cost, cost))

if __name__ == "__main__":
    unittest.main()