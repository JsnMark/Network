# test_layer.py
from layer import *
from activations import ReLu
import unittest
import numpy as np

class TestLayer(unittest.TestCase):
    
    def test_layer_has_correct_attributes(self):
        layer = Layer(2,3, ReLu)
        
        self.assertEqual(layer.num_nodes_in, 2)
        self.assertEqual(layer.num_nodes_out, 3)
        self.assertEqual(layer.activation, ReLu)
        
        
        # Weights matrix (2x3 matrix)
        # let m be row vector
        # [m0]
        # [m1]
        
        self.assertEqual(type(layer.weights), np.ndarray)
        print()
        print(layer.weights)
        print()
        self.assertEqual(len(layer.weights), 2)
        
        
if __name__ == '__main__':
    unittest.main()