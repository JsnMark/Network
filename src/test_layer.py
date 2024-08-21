# test_layer.py
from layer import *
from activations import ReLu, sigmoid, do_nothing
import unittest
import numpy as np

class TestLayer(unittest.TestCase):
    
    def test_layer_has_correct_attributes(self):
        layer = Layer(3,2, ReLu)
        
        self.assertEqual(layer.num_nodes_in, 3)
        self.assertEqual(layer.num_nodes_out, 2)
        self.assertEqual(layer.activation, ReLu)
        
        
        # Weights matrix (2x3 matrix)
        self.assertEqual(type(layer.weights), np.ndarray)
        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(layer.weights[0]), 3)
        self.assertEqual(layer.weights.shape, (2,3))
        
        # Bias vector (2x1)
        self.assertEqual(type(layer.biases), np.ndarray)
        self.assertEqual(len(layer.biases), 2)
        self.assertEqual(layer.biases.shape, (2,))
    
    def test_feed_forward_multiplication_only(self):
        # Act((W)  (i)  + (b))
        # [1 2 3]  [1]    [0]    [14]
        # [4 5 6]  [2]  + [0]  = [32] 
        #          [3]     
        
        layer = Layer(3,2,do_nothing)
        new_weights = np.array([[1,2,3],
                                [4,5,6]])
        
        new_biases = np.array([0,0])
        
        self.assertEqual(new_weights.shape, layer.weights.shape)
        self.assertEqual(new_biases.shape, layer.biases.shape)
        
        layer.weights = new_weights
        layer.biases = new_biases
        
        inputs = np.array([1,2,3])
        expected = np.array([14, 32])
        
        self.assertTrue(np.array_equal(layer.feed_forward(inputs), expected))
        
    def test_feed_forward_addition(self):
        # Act((W)  (i)  + (b))
        # [1 2 3]  [1]    [1]    [15]
        # [4 5 6]  [2]  + [2]  = [34] 
        #          [3]     
        class do_nothing:
            def func(n):
                return n
        
        layer = Layer(3,2,do_nothing)
        new_weights = np.array([[1,2,3],
                                [4,5,6]])
        
        new_biases = np.array([1,2])
        
        self.assertEqual(new_weights.shape, layer.weights.shape)
        self.assertEqual(new_biases.shape, layer.biases.shape)
        
        layer.weights = new_weights
        layer.biases = new_biases
        
        inputs = np.array([1,2,3])
        expected = np.array([15,34])
        
        self.assertTrue(np.array_equal(layer.feed_forward(inputs), expected))
        
    def test_feed_forward_activation(self):
        # sigmoid((W)      (i)  + (b))
        # sigmoid([1 2 3]  [1]    [1])   [0.999999694097773]
        #         [4 5 6]  [2]  + [2]  = [0.9999999999999982] 
        #                  [3]     

        
        layer = Layer(3,2,sigmoid)
        new_weights = np.array([[1,2,3],
                                [4,5,6]])
        
        new_biases = np.array([1,2])
        
        self.assertEqual(new_weights.shape, layer.weights.shape)
        self.assertEqual(new_biases.shape, layer.biases.shape)
        
        layer.weights = new_weights
        layer.biases = new_biases
        
        inputs = np.array([1,2,3])
        expected = np.array([0.999999694097773,0.9999999999999982])
        
        self.assertTrue(np.array_equal(layer.feed_forward(inputs), expected))
         
    def test_cost_of_single_training_example(self):
        inputs = np.array([0,1,1])
        expected_outputs = np.array([1,0])
        
        layer = Layer(3,2,do_nothing)
        weights = np.array([[1,2,3],
                            [-1,-2,-3]])
        biases = np.array([1,2])
        self.assertEqual(weights.shape, layer.weights.shape)
        self.assertEqual(biases.shape, layer.biases.shape)
        layer.weights = weights
        layer.biases = biases
        
        # output = [36, 9]
        # cost = 45
        
        expected_cost = 34
        actual_cost = layer.single_cost(inputs, expected_outputs)
        self.assertEqual(expected_cost, actual_cost)
    
    def test_average_cost(self):
        inputs = [np.array([0,1,1]), np.array([1,1,1])]
        expected_outputs = [np.array([1,0]), np.array([0,1])]
        
        layer = Layer(3,2,do_nothing)
        weights = np.array([[1,2,3],
                            [-1,-2,-3]])
        biases = np.array([1,2])
        self.assertEqual(weights.shape, layer.weights.shape)
        self.assertEqual(biases.shape, layer.biases.shape)
        layer.weights = weights
        layer.biases = biases
        
        expected_costs = [34, 74]
        actual_costs = layer.cost_function(inputs, expected_outputs)
        
        self.assertEqual(sum(expected_costs)/2.0, actual_costs)

        
if __name__ == '__main__':
    unittest.main()