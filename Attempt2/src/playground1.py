#%%
import csv
import activations as act
import layers as layers
import network as net
import matplotlib.pyplot as plt
import data_manager as dm
import numpy as np

# From Harvards cs50 banknotes to read banknotes.csv
with open("data/banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]


np_evidence = dm.to_numpy_array(evidence)
np_labels = np.expand_dims(dm.to_numpy_array(labels), axis=1)

X_training, X_testing, y_training, y_testing = dm.split_data(np_evidence, np_labels, test_size=0.4 )

network = net.Network([4,8,8,1], [act.Sigmoid, act.Sigmoid, act.Sigmoid])

print(f"Original cost: {sum(network.cost_function(X_training, y_training))}")
network.train(X_training, y_training, 0.03, 1000)
print(f"New cost: {sum(network.cost_function(X_training, y_training))}")


yHat = network.classify(X_testing)

print(network.accuracy(y_testing, yHat))
