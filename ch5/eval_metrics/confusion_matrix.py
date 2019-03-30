from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits = load_digits()
y = (digits.target == 9)
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=0)

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
score = confusion_matrix(y_test, logreg.predict(X_test))
print(score)
