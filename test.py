import src.activations as act
import src.layer as layer
import src.network as net
import numpy as np


layers = [2,16,2]
activations = [act.ReLu, act.ReLu]
network = net.Network(layers, activations)

data = [[3, 9], [6, 6], [3, 6], [7, 9], [7, 5], [3, 10], [7, 6], [7, 2], [7, 9], [2, 6], [8, 8], [4, 6], [1, 2], [9, 8], [7, 10], [8, 2], [8, 2], [8, 2], [5, 6], [3, 5], [5, 9], [9, 10], [10, 7], [8, 3], [4, 1], [6, 7], [7, 6], [3, 1], [5, 8], [3, 6], [10, 4], [9, 5], [2, 9], [9, 5], [9, 7], [6, 9], [8, 8], [8, 7], [1, 6], [1, 5], [7, 6], [4, 3], [3, 6], [3, 10], [1, 5], [10, 4], [8, 9], [9, 5], [8, 10], [6, 10]]
safes = [q for q in data if q[1] < 7]

data = [np.array(point) for point in data]

# Anything with a y values of 7+ is poisonous, 6 and under is safe
expected_outputs = [np.array([1,0]) if point[1] < 7 else np.array([0,1]) for point in data]

# print("Weights")
# print(network.layers[-1].weights)
# print(network.layers[-1].biases)

cost = network.cost_function(data, expected_outputs)
# print("Cost")
# print(cost)

network.backpropogation(data, expected_outputs)

cost = network.cost_function(data, expected_outputs)
# print("Cost")
# print(cost)