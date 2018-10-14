from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('agg')  # Needed when you use WSL
import matplotlib.pyplot as plt


# generate test and train data
cancer = load_breast_cancer()
trn_X, test_X, trn_y, test_y = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# prepare the variables for plot
trn_accuracy = []
test_accuracy = []

# try n_neighbors from 1 to 10
neighbors_setting = range(1, 11)

for n_neighbors in neighbors_setting:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(trn_X, trn_y)

    # record the training accuracy
    trn_accuracy.append(clf.score(trn_X, trn_y))
    # record the test accuracy
    test_accuracy.append(clf.score(test_X, test_y))

plt.plot(neighbors_setting, trn_accuracy, label="training accuracy")
plt.plot(neighbors_setting, test_accuracy, label="test accuracy")
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('knn.png')
