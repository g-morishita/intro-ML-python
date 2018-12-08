from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

iris = load_iris()
X_trainval, X_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=0)

gammas = [0.001, 0.01, 0.1, 1, 10, 100]
Cs = [0.001, 0.01, 0.1, 1, 10, 100]

best_score = 0
for gamma in gammas:
    for C in Cs:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_parameters = {'gamma': gamma, 'C': C}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
print(f"Test score: {svm.score(X_test, y_test)}")
print(f"best parameters: {best_parameters}")
