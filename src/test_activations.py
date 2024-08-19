from activations import *
import unittest

class TestActivations(unittest.TestCase):
    
    def test_sigmoid(self):
        x = sigmoid(0.5)
        self.assertAlmostEqual(x, .6224593312018958)
        
        x = sigmoid(-1.3)
        self.assertAlmostEqual(x, .21416501695736453)
        
        x = sigmoid(0)
        self.assertAlmostEqual(x, 0.5)


    def test_ReLu(self):
        x = ReLu(0.5)
        self.assertAlmostEqual(x, .5)
        
        x = ReLu(-1.3)
        self.assertAlmostEqual(x, 0)
        
        x = ReLu(0)
        self.assertAlmostEqual(x, 0)
        
if __name__ == '__main__':
    unittest.main()