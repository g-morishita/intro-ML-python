import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

reviews_train = load_files("data/aclImdb/train", categories=["pos", "neg"])
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = load_files("data/aclImdb/test", categories=["pos", "neg"])
text_test, y_test = reviews_test.data, reviews_test.target

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print(f"X_train:\n{repr(X_train)}")

feature_names = vect.get_feature_names()
print(f"Number of features: {len(feature_names)}")
print(f"First 20 features:\n{X_train[:20]}")
print(f"Features 20010 to 20030:\n{X_train[20010:20030]}")
print(f"Every 2000th feature:\n{X_train[::2000]}")

score = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print(f"Mean cross-validation accuracy: {np.mean(score):.2f}")

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=300), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best cross-validation score: {grid.best_score_:.2f}")
print(f"Best parameters: {grid.best_params_}")

X_test = vect.transform(text_test)
print(f"{grid.score(X_test, y_test):.2f}")
