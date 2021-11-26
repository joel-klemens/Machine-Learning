# Machine-Learning
Practice ML by solving some of the most famous data sets. 

Data sets from Scikit Learn. 

## Solving Iris data set 

This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

This data set is solved using KNN: 
- classification alogrithm where an object is classified based on the classification label of its nearest neighbors (euclidean distance between data point and nearest k neighbors, k being an odd number to avoid ties). The classification is determined based on the majority class of neighbors. 

## Solving Linear Regression 

Generate a random regression problem and solve the approximation by finding the minimum of the mean squared cost function.
Calculate derivative of the gradient with respect to w and b 
Gradient descent is used to iterate in the direction of the steepest descent (defined by negative of the gradient)
Each iteration we update the weight and the bias (w and b) (example: w = w - a * dw) alpha being the learning rate. 



### Resources used for learning 

Iris with KNN:
    - https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
    - https://www.youtube.com/watch?v=ngLyX54e1LU&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=2&ab_channel=PythonEngineer

Linear regression: 
    - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
    - https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
    - https://www.youtube.com/watch?v=4swNt7PiamQ&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=2&ab_channel=PythonEngineer
    - https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html#:~:text=Gradient%20descent%20is%20an%20optimization,the%20parameters%20of%20our%20model.


