#%%
import csv
import activations as act
import layers as layers
import network as net
import matplotlib.pyplot as plt
import data_manager as dm
import numpy as np


def read(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader:
            number = int(row[:1][0])
            label = [1 if i == number else 0 for i in range(10)]
            data.append({
                "evidence": [float(cell) for cell in row[1:]],
                "label": label
            })

    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]
    return evidence, labels

print("Reading tests...")
X_testing, y_testing = read("data/mnist_test.csv")

print("Reading training...")
X_training, y_training = read("data/mnist_train.csv")

print("To numpy...")
X_testing = dm.to_numpy_array(X_testing)
X_training = dm.to_numpy_array(X_training)

y_testing = dm.to_numpy_array(y_testing)
y_training = dm.to_numpy_array(y_training)

print("Normalize inputs...")
X_testing = X_testing / 255.0
X_training = X_training / 255.0

print("Network...")
network = net.Network([784, 16, 16, 10], [act.Sigmoid, act.Sigmoid, act.Sigmoid])

print(f"Initial cost: {sum(network.cost_function(X_training, y_training))}")
yHat = network.feed_forward(X_testing)
print(f"Initial accuracy: {network.accuracy(y_testing, yHat)}")

# network.train(X_training, y_training, 0.003, 100)
network.train_batches(X_training, y_training, 0.01, times=10, batch_size=512)
print(f"New cost: {sum(network.cost_function(X_training, y_training))}")


yHat = network.feed_forward(X_testing)

print(f"Final accuracy: {network.accuracy(y_testing, yHat)}")