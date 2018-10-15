from statistics import mean
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cancer = load_breast_cancer()

    neighbors_setting = range(1, 11)

    training_accuracy = []
    test_accuracy = []

    for n_neighbors in neighbors_setting:
        training_scores = []
        test_scores = []

        for i in range(200):
            train_X, test_X, train_y, test_y = train_test_split(
                cancer.data, cancer.target, stratify=cancer.target)

            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(train_X, train_y)

            training_scores.append(clf.score(train_X, train_y))
            test_scores.append(clf.score(test_X, test_y))

        training_accuracy.append(mean(training_scores))
        test_accuracy.append(mean(test_scores))

    plt.plot(neighbors_setting, training_accuracy, label="training")
    plt.plot(neighbors_setting, test_accuracy, label="test")
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('/mnt/c/Users/optim/Desktop/review.png')
