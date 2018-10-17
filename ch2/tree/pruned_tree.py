from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.target_names)
train_X, test_X, train_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(train_X, train_y)

print(f"Accuracy on training set: {tree.score(train_X, train_y)}")
print(f"Accuracy on test set: {tree.score(test_X, test_y)}")

export_graphviz(tree, out_file="tree.dot", class_names=cancer.target_names,
                feature_names=cancer.feature_names, impurity=False, filled=True)
