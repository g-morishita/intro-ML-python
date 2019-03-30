from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
y = (digits.target == 9)

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

from sklearn.dummy import DummyClassifier
import numpy as np
dummy_majority = DummyClassifier(
    strategy='most_frequent').fit(X_train, y_train)
pred = dummy_majority.predict(X_test)
print(f"Unique prediction: {np.unique(pred)}")
print(f"Dummy Accuracy: {dummy_majority.score(X_test, y_test)}")

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
print(f"Tree Accuracy: {dtree.score(X_test, y_test)}")
