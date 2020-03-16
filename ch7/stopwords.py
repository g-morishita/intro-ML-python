from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

reviews_train = load_files('./data/aclImdb/train')
reviews_test = load_files('./data/aclImdb/test')

text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

vect = CountVectorizer(min_df=5, max_df=3000, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best score is {grid.best_score_}")
