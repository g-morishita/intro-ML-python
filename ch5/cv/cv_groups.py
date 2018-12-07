from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score

X, y = make_blobs(n_samples=12, random_state=0)
logreg = LogisticRegression()
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print(f"Cross-validation scores: \n{scores}")
