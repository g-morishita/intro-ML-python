from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print(f"Confusion Matrix\n: {confusion_matrix(y_test, pred)}")
print(f"Classification report\n: {classification_report(y_test, pred)}")
