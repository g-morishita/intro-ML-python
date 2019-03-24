from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

iris = load_iris()
kfold = KFold(n_splits=3)

X = iris.data
y = iris.target
logreg = LogisticRegression()

scores = cross_val_score(logreg, X, y, cv=kfold)
print(f"Cross-validation scores:\n {scores}")
