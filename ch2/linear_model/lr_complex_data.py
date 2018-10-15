from mglearn.datasets import load_extended_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_extended_boston()
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

lr = LinearRegression()
lr.fit(train_X, train_y)

print(f"Training set score: {lr.score(train_X, train_y):.2f}")
print(f"Training set score: {lr.score(test_X, test_y):.2f}")
