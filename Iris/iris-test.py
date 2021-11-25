import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

color_map = ListedColormap(['b','r','g'])

# get the iris data set from scikit learn 
iris = datasets.load_iris()
X = iris.data
y = iris.target

# sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# random state 1234 to make it reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234) 

print(X_train.shape) # 4 dimensions 
print(X_train[0])   # take a look at the first set of features sepal length in cm, 
                    # sepal width in cm, petal length in cm, petal width in cm

print(y_train.shape) # 1 D row vector for classification Iris-setosa, iris-versicolour, iris-virginica
print(y_train) # print out all class 

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=color_map, edgecolor='k', s=20)
plt.show() 
