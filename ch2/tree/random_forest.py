from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

rf = RandomForestClassifier(n_estimators=1000, random_state=0)
rf.fit(X_train, y_train)

print(f"Accuracy on traning set: {rf.score(X_train, y_train)}")
print(f"Accuracy on test set: {rf.score(X_test, y_test)}")

n_features = cancer.data.shape[1]

plt.barh(range(n_features), rf.feature_importances_, align='center')
plt.yticks(range(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.savefig("Random_forest_importance.png", bbox_inches='tight')
