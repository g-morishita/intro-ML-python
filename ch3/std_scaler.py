import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# using the entire data to scale
scaler = StandardScaler()
scaler.fit(cancer.data)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc = SVC(C=100)
svc.fit(X_train, y_train)
print(f"Accuracy without scaling: {svc.score(X_test, y_test)}")

svc.fit(X_train_scaled, y_train)
print(f"Accuracy using entire data: {svc.score(X_test_scaled, y_test)}")

# using the training data only to scale
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svc.fit(X_train_scaled, y_train)
print(f"Accuracy using training data: {svc.score(X_test_scaled, y_test)}")
