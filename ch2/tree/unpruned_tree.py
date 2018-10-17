from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
train_X, test_X, train_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(train_X, train_y)
print(f"Accuracy on training set: {tree.score(train_X, train_y)}")
print(f"Accuracy on test set: {tree.score(test_X, test_y)}")
