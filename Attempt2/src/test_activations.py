import unittest
from activations import *

class test_sigmoid(unittest.TestCase):
    def test_single_input_function(self):
        x = 4
        yHat = Sigmoid.function(x)
        y = 0.98201379003791
        self.assertEqual(type(yHat), np.float64 )
        self.assertAlmostEqual(yHat, y)
        
    def test_1d_array_function(self):
        x1 = np.array([1,2])
        yHat1 = Sigmoid.function(x1)
        y1 = np.array([0.73105857863, 0.88079707797788])
       
        self.assertEqual(type(yHat1), np.ndarray)
        self.assertTrue(np.allclose(yHat1, y1))
        
        x2 = np.array([[1],
                      [2]])
        yHat2 = Sigmoid.function(x2)
        y2 = np.array([[0.73105857863],
                      [0.88079707797788]])
       
        self.assertEqual(type(yHat2), np.ndarray)
        self.assertTrue(np.allclose(yHat2, y2)) 
        
    def test_2d_array_function(self):
        x = np.array([[1,2],
                      [3,4],
                      [5,6]])
        yhat = Sigmoid.function(x)
        y = np.array([[ 0.73105857863, 0.88079707797788],
                      [0.95257412682243, 0.98201379003791],
                      [ 0.99330714907572, 0.99752737684337]])
        self.assertEqual(type(yhat), np.ndarray)
        self.assertEqual(type(yhat[0]), np.ndarray)
        self.assertTrue(np.allclose(yhat, y))
        
        

if __name__ == "__main__":
    unittest.main()