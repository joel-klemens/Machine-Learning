import numpy as np

def euclid_dist(x1, x2):
    # return the euclidean distance between two points 
    return np.sqrt(np.sum((x1 - x2) ** 2))