from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# preprocessing
mean_on_training = X_train.mean(axis=0)
std_on_training = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_training) / std_on_training
X_test_scaled = (X_test - mean_on_training) / std_on_training

mlp = MLPClassifier(max_iter=1000, alpha=1, verbose=True, random_state=0)
mlp.fit(X_train_scaled, y_train)

print(f"Accuracy on training: {mlp.score(X_train_scaled, y_train)}")
print(f"Accuracy on test: {mlp.score(X_test_scaled, y_test)}")
