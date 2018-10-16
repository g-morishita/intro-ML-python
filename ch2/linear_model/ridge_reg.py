import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mglearn.datasets import load_extended_boston
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_extended_boston()
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

ridge = Ridge(alpha=1.0)
ridge.fit(train_X, train_y)

ridge10 = Ridge(alpha=10.0)
ridge10.fit(train_X, train_y)

ridge01 = Ridge(alpha=0.1)
ridge01.fit(train_X, train_y)

lr = LinearRegression()
lr.fit(train_X, train_y)

train_scores = []
test_scores = []
alphas = ['1', '10', '0.1']

train_scores.append(ridge.score(train_X, train_y))
train_scores.append(ridge10.score(train_X, train_y))
train_scores.append(ridge01.score(train_X, train_y))

test_scores.append(ridge.score(test_X, test_y))
test_scores.append(ridge10.score(test_X, test_y))
test_scores.append(ridge01.score(test_X, test_y))

fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].bar(alphas, train_scores, align="center")
axes[0].set_title("Training")
axes[1].bar(alphas, test_scores, align="center")
axes[1].set_title("Test")
fig.savefig("ridge.png")
plt.close()

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge01.coef_, '^', label="Ridge alpha=0.1")
plt.plot(ridge10.coef_, 'v', label="Ridge alpha=10")
plt.plot(lr.coef_, 'o', label="Linear Regression")

plt.xlabel("Coefficient Index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.savefig("ridge_scatter.png")
