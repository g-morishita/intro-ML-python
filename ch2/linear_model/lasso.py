import numpy as np
from mglearn.datasets import load_extended_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = load_extended_boston()
train_X, test_X, train_y, test_y = train_test_split(X, y)

lasso = Lasso(alpha=0.01, max_iter=100_000)
lasso.fit(train_X, train_y)

print(f"Training set score: {lasso.score(train_X, train_y)}")
print(f"Test set score: {lasso.score(test_X, test_y)}")
print(f"The number of unused features: {np.sum(lasso.coef_ == 0)}")
print(f"The number of used features: {np.sum(lasso.coef_ != 0)}")
