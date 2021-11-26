import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from make_regression import LinearRegression

# get the random regression problem 
# sklearn.datasets.make_regression(n_samples=100, n_features=100, *, n_informative=10, n_targets=1, 
# bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
X, y = datasets.make_regression(n_samples=200, n_features=1, noise=5, random_state=5)

# sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
# random state to make it reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=555)

#fig = plt.figure(figsize=(8,6))
#plt.scatter(X[:,0], y, color="b", marker="o", s = 30)
#plt.show() 

# take a look at the data
#print(X_train.shape)
#print(X_train[0])
#print(y_train.shape)
#print(y_train)


# test the solution to the data set, increasing learning rate and number of iterations will yeild better results 
regressor = LinearRegression(learning_rate = 0.007)
# fit samples and labels 
regressor.fit(X_train, y_train)
# predicted values 
predicted = regressor.predict(X_test) 

# cost function -> to tell us how big the real vs approx is 
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = mse(y_test, predicted)
print("Cost, lower is better: ", mse_value)

# plot the linear prediction line, test data and training data 
fig = plt.figure(figsize=(8,6))
prediction_line = regressor.predict(X) 
training_points = plt.scatter(X_train, y_train, color="b", s=10)
testing_points = plt.scatter(X_test, y_test, color="r", s=10)
plt.plot(X,prediction_line, color="black")
plt.show() 
