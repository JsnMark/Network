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
         
if __name__ == '__main__':
    unittest.main()