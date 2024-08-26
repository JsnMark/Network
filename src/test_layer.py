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
        self.assertIsInstance(layer.activation_function, ReLu)
        self.assertEqual(layer.a_n, None)
        self.assertEqual(layer.a_m, None)
        
        # Weights matrix (2x3 matrix)
        self.assertEqual(type(layer.weights), np.ndarray)
        self.assertEqual(len(layer.weights), 2)
        self.assertEqual(len(layer.weights[0]), 3)
        self.assertEqual(layer.weights.shape, (2,3))
        
        self.assertEqual(type(layer.weight_gradient_matrix), np.ndarray)
        self.assertEqual(len(layer.weight_gradient_matrix), 2)
        self.assertEqual(len(layer.weight_gradient_matrix[0]), 3)
        self.assertEqual(layer.weight_gradient_matrix.shape, (2,3))
        
        # Bias vector (2x1)
        self.assertEqual(type(layer.biases), np.ndarray)
        self.assertEqual(len(layer.biases), 2)
        self.assertEqual(layer.biases.shape, (2,))
        
        self.assertEqual(type(layer.bias_gradient), np.ndarray)
        self.assertEqual(len(layer.bias_gradient), 2)
        self.assertEqual(layer.bias_gradient.shape, (2,))
    
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
        self.assertTrue(np.array_equal(layer.a_m, inputs))
        self.assertTrue(np.array_equal(layer.a_n, expected))
        
    def test_feed_forward_addition(self):
        # Act((W)  (i)  + (b))
        # [1 2 3]  [1]    [1]    [15]
        # [4 5 6]  [2]  + [2]  = [34] 
        #          [3]     
        
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
        actual_output = layer.feed_forward(inputs)
        cost = layer.single_cost(actual_output, expected_outputs)
        self.assertEqual(expected_cost, cost)
    

    def test_calculate_gradient(self):
        layer = Layer(3,2,ReLu)
        weight = np.array([[1.0,2.0,3.0],
                           [4.0,5.0,6.0]])
        bias = np.array([0.0,0.0])
        
        input = np.array([1.0,1.0,1.0])
        expected_output = np.array([1.0,0.0])
        
        layer.weights = weight
        layer.biases = bias
        
        actual_output = layer.feed_forward(input)
        
        print(layer.single_cost(actual_output, expected_output))
        
        layer.calculate_output_gradient(input, actual_output, expected_output)
        
        

        
if __name__ == '__main__':
    unittest.main()