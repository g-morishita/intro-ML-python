import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# read data
iris = load_iris()
X, y = iris.data, iris.target

# instantiate a Grid Search instance
params = {'C': np.geomspace(0.001, 100, 6),
          'gamma': np.geomspace(0.001, 100, 6)}
grid_search = GridSearchCV(SVC(), params, cv=5)

# split the data into a training and a test set
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

grid_search.fit(train_X, train_y)

print(f"Best Param: {grid_search.best_params_}")
print(f"Cross Validation Score: {grid_search.best_score_:.2f}")
print(f"Test set Score: {grid_search.score(test_X, test_y):.2f}")
