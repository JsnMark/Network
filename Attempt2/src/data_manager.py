# data_manager.py
import numpy as np

def to_numpy_array(sequence, dtype="float64"):
    '''Creates a numpy array of a given sequence'''
    return np.array(sequence, dtype=dtype)

def normalize_columns(arr):
    '''Normalizes column data given a numpy array'''
    return arr / arr.max(axis=0)

def split_data(evidence, labels, test_size=0.5):
    '''Splits the data given to training and testing'''
    
    concat = np.concatenate((evidence, labels), axis=1)
    np.random.shuffle(concat)

    length = len(evidence)
    
    test_amount = round(length * test_size)
    
    testing = concat[:test_amount]
    training = concat[test_amount:]
    
    X_training = training[test_amount:, :-1]
    X_testing = testing[:test_amount, :-1]
    
    y_training = training[test_amount:, -1:]
    y_testing = testing[:test_amount, -1:]
    
    return X_training, X_testing, y_training, y_testing
    
