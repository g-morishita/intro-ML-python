from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)

# transofrm data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transfor(X_test)

# check if transformation works
# print(f"transformed shape: {X_train_scaled.shape}")
# print(
#     f"per-feature min before scaling: {np.round(X_train.min(axis=0), decimals=2)}")
# print(
#     f"per-feature max before scaling: {np.round(X_train.max(axis=0), decimals=2)}")
# print(f"per-feature min after scaling: {X_train_scaled.min(axis=0)}")
# print(f"per-feature max after scaling: {X_train_scaled.max(axis=0)}")
