# test_network.py
from network import *
from layer import *
from activations import ReLu, sigmoid, do_nothing
import unittest
import numpy as np

class TestNetwork(unittest.TestCase):
    
    def test_network_has_correct_attributes(self):
        layers = [2,3,4,2]
        activations =  [ReLu, sigmoid, sigmoid]
        
        network = Network(layers, activations)
        
        self.assertEqual(len(network.layers), 3)
        self.assertIsInstance(network.layers[0].activation, ReLu)
        self.assertEqual(network.layers[0].num_nodes_in, 2)
        self.assertEqual(network.layers[0].num_nodes_out, 3)
        
        self.assertIsInstance(network.layers[1].activation, sigmoid)
        self.assertEqual(network.layers[1].num_nodes_in, 3)
        self.assertEqual(network.layers[1].num_nodes_out, 4)
        
        self.assertIsInstance(network.layers[2].activation, sigmoid)
        self.assertEqual(network.layers[2].num_nodes_in, 4)
        self.assertEqual(network.layers[2].num_nodes_out, 2)
    
    def test_calculate_output_normal_case(self):
        layers = [2,3,2]
        activations = [do_nothing, sigmoid]
        
        # (3x2)(2x1) = (3x1)
        # (2x3)(3x1) = (2x1) 
        
        weight_1 = np.array([[1,1],
                            [2,2],
                            [3,3]])
        
        bias_1 = np.array([10, 9, 8])
        
        weight_2 = np.array([[-3, 2, 1], [-3, 2, 1.5]])
        
        bias_2 = np.array([1, 2])
        
        input = [1, 0]
        
        # Calculated outside of this program
        expected_output = np.array([0.731058578630074, 0.9994472213630777])
        
        network = Network(layers, activations)
        
        self.assertEqual(weight_1.shape, network.layers[0].weights.shape)
        self.assertEqual(bias_1.shape, network.layers[0].biases.shape)
        self.assertEqual(weight_2.shape, network.layers[1].weights.shape)
        self.assertEqual(bias_2.shape, network.layers[1].biases.shape)
        
        network.layers[0].weights = weight_1
        network.layers[0].biases = bias_1
        network.layers[1].weights = weight_2
        network.layers[1].biases = bias_2
        
        self.assertTrue(np.allclose(network.calculate_output(input), expected_output))
        
    def test_average_cost(self):
        inputs = [np.array([0,1,1]), np.array([1,1,1])]
        expected_outputs = [np.array([1,0]), np.array([0,1])]
        
        network = Network([3,2], [do_nothing])
        
        weights = np.array([[1,2,3],
                            [-1,-2,-3]])
        biases = np.array([1,2])
        self.assertEqual(weights.shape, network.layers[0].weights.shape)
        self.assertEqual(biases.shape, network.layers[0].biases.shape)
        network.layers[0].weights = weights
        network.layers[0].biases = biases
        
        expected_costs = [34, 74]
        actual_costs = network.cost_function(inputs, expected_outputs)
        
        self.assertEqual(sum(expected_costs)/2.0, actual_costs)
        
if __name__ == '__main__':
    unittest.main()