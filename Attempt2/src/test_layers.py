# test_layers.py
import unittest
import numpy as np
from activations import *
from layers import *

class LayersTest(unittest.TestCase):
    def test_correct_parameters(self):
        l = Layer(2, 3, Sigmoid)
        self.assertEqual(l.W.shape, (2,3))
        self.assertEqual(l.b.shape, (3,))
        
    def test_set_weights(self):
        layer = Layer(2,3, Sigmoid)
        correct_w = np.array(([[1,2,2],
                               [1,2,1]]))
        layer.set_weights(correct_w)
        self.assertEqual(layer.W.shape,(2,3))
        
        incorrect_w = np.array([[1,2],
                                [3,4],
                                [4,5]])
        with self.assertRaises(Exception):
            layer.set_weights(incorrect_w)
    
    def test_set_weights(self):
        layer = Layer(2,3, Sigmoid)
        correct_b = np.array(([1,2,2]))
        layer.set_biases(correct_b)
        self.assertEqual(layer.b.shape,(3,))
        
        incorrect_b = np.array([1,2])
        with self.assertRaises(Exception):
            layer.set_biases(incorrect_b)
        
        
    def test_forward_correct_size(self):
        # (4x2)(2x3) == (4x3)
        l = Layer(2,3, Sigmoid)
        inputs = np.array([[1,1],
                           [1,1],
                           [1,1],
                           [1,1]])
        outputs = l.forward(inputs)
        self.assertEqual(outputs.shape, (4,3))
        self.assertEqual(l.X.shape, (4,2))
        self.assertEqual(l.z.shape, (4,3))
        self.assertEqual(l.a.shape, (4,3))
        
    def test_forward_correct_output(self):
        layer = Layer(2,3, Sigmoid)
        inputs = np.array([[1,2],
                           [2,1],
                           [3,1],
                           [1,3]])
        W = np.array([[0,1,2],
                      [3,4,5]])
        b = np.array([0,1,2])
        self.assertEqual(W.shape, layer.W.shape)
        self.assertEqual(b.shape, layer.b.shape)
        layer.W = W
        layer.b = b
        
        # calculated by hand. used to compare to layer.z
        z = np.array([[6, 10, 14],
                      [3, 7, 11],
                      [3, 8, 13],
                      [9,14,19]])
        
        output = layer.forward(inputs)
        self.assertTrue(np.array_equal(layer.z, z))
        self.assertEqual(output.shape, (4,3))
        # We know Sigmoid function works on matrix from test_activations.py
        

        

        
        
if __name__ == "__main__":
    unittest.main()