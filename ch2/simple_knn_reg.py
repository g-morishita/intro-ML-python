from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=40)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(train_X, train_y)
print(f"The prediction is {clf.predict(test_X)}")
print(f"The score is {clf.score(test_X, test_y)}")
