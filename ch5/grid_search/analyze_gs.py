from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from pprint import pprint

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
grid_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(), grid_params, cv=5, return_train_score=False)
grid_search.fit(X_train, y_train)
results = pd.DataFrame(grid_search.cv_results_)
pprint(results)
