import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import mglearn as mg
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mg.datasets.make_wave(n_samples=40)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 5], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(train_X, train_y)

    ax.plot(line, reg.predict(line))
    ax.plot(train_X, train_y, '^', c=mg.cm2(0), markersize=8)
    ax.plot(test_X, test_y, 'v', c=mg.cm2(1), markersize=8)

    ax.set_title(
        f"{n_neighbors} neighbors\n train score: {reg.score(train_X, train_y):.2f} test score: {reg.score(test_X, test_y):.2f}")
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')

axes[0].legend(["Model predictions", "Training data/target",
                "est data/target"], loc="best")
fig.savefig('/mnt/c/Users/optim/Desktop/aa.png')
