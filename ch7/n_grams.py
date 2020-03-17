from sklearn.datasets import load_files
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

reviews_train = load_files('data/aclImdb/train', categories=['pos', 'neg'])
reviews_test = load_files('data/aclImdb/test', categories=['pos', 'neg'])
text_train, y_train = reviews_train.data, reviews_train.target
text_test, y_test = reviews_test.data, reviews_test.target

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression(max_iter=1000))
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10], 
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1,3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print(f"Best Score: {gird.best_score_}")
print(f"Best Param: {grid.best_param_}")
