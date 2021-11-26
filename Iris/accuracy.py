import numpy as np

# helper function to determine accuracy of model 
def accuracy(actual, predicted):
    accuracy = np.sum(actual == predicted) / len(actual)
    return accuracy