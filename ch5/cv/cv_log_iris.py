from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# read data
iris = load_iris()
logreg = LogisticRegression()
print(iris.target)

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print(f"Cross-Validation scores: {scores}")
