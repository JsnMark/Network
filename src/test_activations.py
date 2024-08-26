from activations import *
import unittest

class TestActivations(unittest.TestCase):
    
    def test_sigmoid(self):
        f = sigmoid()
        x = f.func(0.5)
        self.assertAlmostEqual(x, .6224593312018958)
        
        x = f.func(-1.3)
        self.assertAlmostEqual(x, .21416501695736453)
        
        x = f.func(0)
        self.assertAlmostEqual(x, 0.5)

    def test_ReLu(self):
        f = ReLu()
        x = f.func(0.5)
        self.assertAlmostEqual(x, .5)
        
        x = f.func(-1.3)
        self.assertAlmostEqual(x, 0)
        
        x = f.func(0)
        self.assertAlmostEqual(x, 0)
        
if __name__ == '__main__':
    unittest.main()