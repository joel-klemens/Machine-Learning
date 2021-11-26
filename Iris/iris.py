import numpy as np
from collections import Counter
from euclidean_dist import euclid_dist 

class KNN:
    def __init__(self, k=3):
        # k = number of nearest neighbors 
        self.k = k

    def fit(self, X, y):
        # KNN does not need a training step, store samples 
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # predict the classification labels 
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        ## helper method for each single sample  
        # get distances between x and all samples in the dataset 
        distances = [euclid_dist(x, x_train) for x_train in self.X_train]

        # get k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]

        # get the class labels of the NN's 
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # make majority vote for predicted classification 
        most_common = Counter(k_nearest_labels).most_common(1)
        # return the first item from the touple 
        return most_common[0][0]