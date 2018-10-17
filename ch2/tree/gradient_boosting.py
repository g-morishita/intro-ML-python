from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(max_depth=1, random_state=0)
gbrt.fit(X_train, y_train)

n_features = cancer.data.shape[1]
plt.barh(range(n_features), gbrt.feature_importances_)
plt.yticks(range(n_features), cancer.feature_names)
plt.xlabel('Feature Importanc')
plt.ylabel('Feature')
plt.savefig('Gradient_boosted_importance.png', bbox_inches='tight')
