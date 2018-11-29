import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray')
plt.ylabel("Nubmer of appearance")
plt.xlabel("Value")
plt.savefig("./ho.png")

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print(f"Test Score: {score:.3f}")
