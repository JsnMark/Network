# network.py

import numpy as np
import activations
import layers

class Network:
    def __init__(self, layer_sizes: list[int], activation_list: list[type]):
        if len(layer_sizes) != len(activation_list) + 1:
            raise Exception("length of layers must be one more than length of activations")
        
        self.layers = [layers.Layer(layer_sizes[i], layer_sizes[i+1], activation_list[i]) for i in range(len(layer_sizes) - 1)]
    
    def feed_forward(self, X):
        '''Takes input X and feeds it forward through the network to reach an output'''
        input = X
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def cost_function(self, X, y):
        '''Takes in an input (X) and its expected values (y) and returns the cost'''
        outputs = self.feed_forward(X) # yhat, ixn
        # y(ixn) - output(ixn) = ixn
        # This new ixn matrix then has each term squared
        # Then its columns are summed to get an n sized row vector
        # Then each term is halved (still n sized vector)
        c = 0.5 * sum((y - outputs) ** 2) # cost, (size n)
        return c
    
    def gradient_descent(self, X, y):
        '''Performs gradient descent, returns the weight and bias gradients in reversed order(output to input)'''

        weight_gradients = []
        bias_gradients = []

        yHat = self.feed_forward(X)
        # dCdW = derivative of cost with respect to weight
        # dCdA = derivatve of cost with respect to activation
        # dAdZ = derivative of activation with respect to preactivation (z)
        # dZdW = derivative of z with respect to weight
        
        # dCdW = dCdA * dAdZ * dZdW
        # dBdW = dCdA * dAdZ * dZdB
        
        # Calculate output layer gradients

        output_layer = self.layers[-1]
        # dCdA * dAdZ == delta (dCdZ)
        delta = -(y - yHat) * output_layer.activation.derivative(output_layer.z) # ixn for output layer
        
        dCdW = np.dot(output_layer.X.T, delta)  # mxn for outputlayer, same size as its weight matrix
        dCdB = np.mean(delta, axis=0) # 1xn for output layer, same size as its bias vector
        
        weight_gradients.append(dCdW)
        bias_gradients.append(dCdB)
        
        # Since dC/dA is different for the non output layers, we must calculate it differently if we are going to use it
        # Let Am be the previous layer and An be the current layer (The delta above uses An)
        # dCdAm = dC/dAn * dAn/dZ * dZ/dAm
        #       = delta * dZ/dAm
        dCdAm = np.matmul(delta, output_layer.W.T) # ixm
        
        for i in  reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[i]
            # dCdW = dCdA * dAdZ * dZdW
            # dBdW = dCdA * dAdZ * dZdB
            dCdAn = dCdAm # now that we changed layers, it is in layer n (current) instead of layer m (previous)
            delta  = dCdAn * current_layer.activation.derivative(current_layer.z) # ixn

            dCdW = np.dot(current_layer.X.T, delta) # mxn
            dCdB = np.mean(delta, axis=0) # 1xn
            
            dCdAm = np.matmul(delta, current_layer.W.T) # ixm
            
            weight_gradients.append(dCdW)
            bias_gradients.append(dCdB)

        return weight_gradients, bias_gradients
    
    def update_gradients(self, weight_gradients, bias_gradients, learning_rate):
        length = len(self.layers)
        for i in range(length):
            self.layers[length - i - 1].W -= weight_gradients[i] * learning_rate
            self.layers[length - i - 1].b -= bias_gradients[i] * learning_rate

    def train(self, X, y, learning_rate, times):
        for _ in range(times):
            W, B = self.gradient_descent(X,y)
            self.update_gradients(W, B, learning_rate)
            
    def train_batches(self, X, y, learning_rate, times, batch_size):
        x_len = len(X)
        # each time
        for i in range(times):
            # each batch
            print(f"Iterations: {i}")
            for j in range(0, x_len, batch_size):
                W,B = self.gradient_descent(X[j: j + batch_size], 
                                            y[j: j + batch_size])
                self.update_gradients(W, B, learning_rate)
                
                
        
    def classify(self, X):
        output_layer = self.layers[-1]
        yHat = self.feed_forward(X)
        ret_vals = []
        # if outputlayer has 1 node, then anything below 0.5 is 0, anything above it is 1
        if output_layer.num_nodes_out == 1:
            for p in yHat:
                if p < 0.5:
                    ret_vals.append(0)
                else:
                    ret_vals.append(1)

            ret_vals = np.expand_dims(np.array(ret_vals), axis=1)
            return ret_vals
        
        # else, classify based on node
        else:
            for p in yHat:
                pass
        
    def accuracy(self, y, yHat):
        total = 0.0
        correct = 0.0
        for z in zip(y, yHat):
            total += 1.0
            # if np.array_equal(z[0], z[1]):
            #     correct += 1.0
            y_max = max(z[0])
            yHat_max = max(z[1])
            
            y_index = np.where(z[0] == y_max)[0][0]
            yHat_index = np.where(z[1] == yHat_max)[0][0]
            if y_index == yHat_index:
                correct += 1.0
        return correct, total, correct / total

