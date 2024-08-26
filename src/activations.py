# activations.py
from math import e


class sigmoid:
    def func(self, n: float) -> float:
        return 1.0 / ( 1 + e ** (-n))
    
    def derivative(self, n: float) -> float:
        return self.func(n) * (1.0 - self.func(n))

class ReLu:
    def func(self, n: float) -> float:
        return max(0.0, n)
    
    def derivative(self, n: float) -> float:
        if n < 0: 
            return 0
        else:
            return 1
    
# used for testing
class do_nothing:
    def func(self, n):
        return n
    
    def derivative(self, n):
        return 1