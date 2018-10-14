from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_forge

# generate dataset
X, y = make_forge()
trn_X, test_X, trn_y, test_y = train_test_split(X, y, random_state=1)

# model
for n in [1, 3, 5, 7, 10]:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(trn_X, trn_y)

    # predict
    pred = clf.predict(test_X)
    print(f"The prediction is {pred}")

    # score
    print(f"The score of {n} is {clf.score(test_X, test_y)}")
