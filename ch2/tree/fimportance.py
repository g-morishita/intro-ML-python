from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.target_names)
train_X, test_X, train_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(train_X, train_y)

print(f"Feature Importance\n {tree.feature_importances_}")


def plot_feature_importances_cancer(model):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import numpy as np
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.savefig("tree_feture_importance_cancer.png", bbox_inches='tight')


plot_feature_importances_cancer(tree)
