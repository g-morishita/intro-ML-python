import mglearn as mg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mg.datasets.make_wave(n_samples=60)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

lr = LinearRegression()
lr.fit(train_X, train_y)
print(f"The coefficient is {lr.coef_}")
print(f"The offset is {lr.intercept_}")

print(f"Training set score: {lr.score(train_X, train_y):.2f}")
print(f"Training set score: {lr.score(test_X, test_y):.2f}")
