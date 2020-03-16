import numpy as np
from sklearn.datasets import load_files

category_files = ['pos', 'neg']
reviews_train = load_files("data/aclImdb/train/", categories=category_files)
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
print(f"type of text_train: {type(text_train)}")
print(f"length of text_train: {len(text_train}")
print(f"text_train[1]:\n {text_train[1]}")

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print(f"Samples per class (training): {np.bincount(y_train)}")

reviews_test = load_files("data/aclImbd/train/")
text_test, y_test = reviews_test.data, reviews_test.target
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
print(f"Samples per class (test): {np.bincount(y_test)}")
