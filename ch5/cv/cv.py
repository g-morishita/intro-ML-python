from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def my_cv_score(model, X, y, cv=3):
    import numpy as np
    from numpy.random import permutation
    num = len(y)
    n_fold = int(num / cv)
    perm = permutation(num)
    scores = []

    for i in range(cv):
        test_index = perm[n_fold * i:n_fold * (i+1)]
        train_index = np.setdiff1d(perm, test_index)
        scores.append(model.fit(X[train_index], y[train_index]).score(
            X[test_index], y[test_index]))
    return scores


iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
my_scores = my_cv_score(logreg, iris.data, iris.target, cv=5)
print(scores)
print(my_scores)
