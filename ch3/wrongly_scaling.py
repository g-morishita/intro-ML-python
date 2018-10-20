import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# make and split synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c='blue', label="Training Set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1],
                c='red', label="Test Set", s=60)
axes[0].legend(loc="upper left")
axes[0].set_title("Original Data")

scaler = MinMaxScaler()
scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c='blue', label="Training Set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
                c='red', label="Test Set", s=60)
axes[1].legend(loc="upper left")
axes[1].set_title("Transfromed Data")

fig.savefig("scale.png")
