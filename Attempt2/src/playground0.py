#%%

import activations as act
import layers as layers
import network as net
import numpy as np
import matplotlib.pyplot as plt


layers_ls = [2, 16, 2]
activations = [act.Sigmoid, act.Sigmoid]
network = net.Network(layers_ls, activations)

data = [[3.0, 9.0], [6.0, 6.0], [3.0, 6.0], [7.0, 9.0], [7.0, 5.0], [3.0, 10.0], [7.0, 6.0], [7.0, 2.0], [7.0, 9.0], [2.0, 6.0], [8.0, 8.0], [4.0, 6.0], [1.0, 2.0], [9.0, 8.0], [7.0, 10.0], [8.0, 2.0], [8.0, 2.0], [8.0, 2.0], [5.0, 6.0], [3.0, 5.0], [5.0, 9.0], [9.0, 10.0], [10.0, 7.0], [8.0, 3.0], [4.0, 1.0], [6.0, 7.0], [7.0, 6.0], [3.0, 1.0], [5.0, 8.0], [3.0, 6.0], [10.0, 4.0], [9.0, 5.0], [2.0, 9.0], [9.0, 5.0], [9.0, 7.0], [6.0, 9.0], [8.0, 8.0], [8.0, 7.0], [1.0, 6.0], [1.0, 5.0], [7.0, 6.0], [4.0, 3.0], [3.0, 6.0], [3.0, 10.0], [1.0, 5.0], [10.0, 4.0], [8.0, 9.0], [9.0, 5.0], [8.0, 10.0], [6.0, 10.0]]

# curve
safes = [[3.0, 1.0], [4.0, 1.0], [4.0, 3.0], [4.0, 6.0], [5.0, 6.0], [5.0, 8.0], [6.0, 6.0], [6.0, 7.0], [7.0, 2.0], [7.0, 5.0], [7.0, 6.0], [8.0, 2.0], [8.0, 3.0]]

# diag
# safes = [[1.0, 2.0], [1.0, 5.0], [1.0, 6.0], [2.0, 6.0], [3.0, 1.0], [3.0, 5.0], [3.0, 6.0], [4.0, 1.0], [4.0, 3.0]]

# horizontal
# safes = [q for q in data if q[1] < 7]

# What it should look like
fig, ax = plt.subplots()
for point in data:
    x = point[0]
    y = point[1]
    if point in safes:
        ax.scatter(x, y, marker=".",c="b")
    else:
        ax.scatter(x, y, marker=".",c="r")


np_data = np.array(data)
np_safes = np.array(safes)

norm_np_data = np_data / 10
norm_np_safes = np_safes / 10

np_expected_outputs = np.array([[1.0, 0.0] if point in safes 
                                else [0.0, 1.0] for point in data])

outputs = network.feed_forward(norm_np_data)

print(f"Original cost: {sum(network.cost_function(norm_np_data, np_expected_outputs)) / 2.0}")

network.train(norm_np_data, np_expected_outputs, 0.1, 10000)

print(f"New cost: {sum(network.cost_function(norm_np_data, np_expected_outputs)) / 2.0}")

# Time to classify stuff
# Plot data
fig, ax = plt.subplots()
for point in norm_np_data:
    output = network.feed_forward(point)
    max_val = max(output)
    index = list(output).index(max_val)
    
    
    
    point = [point[0] * 10, point[1] * 10]
    x = point[0]
    y = point[1]
    # Safe
    if index == 0:
        ax.scatter(x, y, marker=".",c="b")
    else:
        ax.scatter(x, y, marker=".",c="r")
plt.xlim(-0, 10.5)  
plt.ylim(-0, 10.5) 
plt.show()