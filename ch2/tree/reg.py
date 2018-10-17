import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ram_prices = pd.read_csv('./mglearn/data/ram_price.csv')

train_data = ram_prices[ram_prices.date < 2000]
test_data = ram_prices[ram_prices.date >= 2000]

X_train = train_data.date[:, np.newaxis]
y_train = np.log(train_data.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(train_data.date, train_data.price, label="Training data")
plt.semilogy(test_data.date, test_data.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()
plt.savefig("reg_tree.png")
