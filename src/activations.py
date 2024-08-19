# activations.py
from math import e

def sigmoid(n: float) -> float:
    return 1.0 / ( 1 + e ** (-n))

def ReLu(n: float) -> float:
    return max(0.0, n)