from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pprint import pprint

# read the data of iris 
iris_dataset = load_iris()
print(f'iris_dataset\'s type is {type(iris_dataset)}') # Bunch object which is similar to dictionary

print(iris_dataset.keys())
# print(iris_dataset['DESCR']) # description of dataset

# split the data into train and test
trn_X, test_X, trn_y, test_y = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# create a model and learn
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(trn_X, trn_y)

# predict
pred = knn.predict(test_X)
pprint(pred)

# score
print(f"The score is {knn.score(test_X, test_y)}")
