# activations.py
from math import e


class sigmoid:
    def func(n: float) -> float:
        return 1.0 / ( 1 + e ** (-n))

class ReLu:
    def func(n: float) -> float:
        return max(0.0, n)
    
# used for testing
class do_nothing:
    def func(n):
        return n