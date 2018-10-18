from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# scale
min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
scaled_X_train = (X_train - min_on_training) / range_on_training

# check if the scaling works properly
# print(f"Minimum for each feature\n {scaled_X_train.min(axis=0)}")
# print(f"Maximum for each feature\n {scaled_X_train.max(axis=0)}")

# scale the test data using the training range and min
scaled_X_test = (X_test - min_on_training) / range_on_training

# learn
svc = SVC(C=1000)
svc.fit(scaled_X_train, y_train)

print(f"Accuracy on training set: {svc.score(scaled_X_train, y_train)}")
print(f"Accuracy on test set: {svc.score(scaled_X_test, y_test)}")
