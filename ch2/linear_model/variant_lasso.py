import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from mglearn.datasets import load_extended_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = load_extended_boston()
train_X, test_X, train_y, test_y = train_test_split(X, y)

lasso001 = Lasso(alpha=0.01, max_iter=100_000)
lasso001.fit(train_X, train_y)

lasso = Lasso(alpha=0.01, max_iter=100_000)
lasso.fit(train_X, train_y)

lasso00001 = Lasso(alpha=0.0001, max_iter=100_000)
lasso00001.fit(train_X, train_y)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.hlines(0, 0, len(lasso.coef_))
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.savefig("scatter_ridge.png")
