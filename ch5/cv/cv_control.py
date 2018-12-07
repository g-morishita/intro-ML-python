from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

iris = load_iris()
logreg = LogisticRegression()

Kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print(
    f"Cross-validation scores: \n{cross_val_score(logreg, iris.data, iris.target, cv=Kfold)}")
