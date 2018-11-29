from sklearn.svm import SVR
import numpy as np
from mglearn.datasets import make_wave
import matplotlib.pyplot as plt

X, y = make_wave(n_samples=100)
line = np.linspace(-3, 3, 100).reshape(-1, 1)

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label=f"SVR gamma={gamma}")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input Feature")
plt.legend(loc="best")
plt.savefig("./hoge.png")
