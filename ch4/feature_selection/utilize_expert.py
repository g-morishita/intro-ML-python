import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

citibike = mglearn.datasets.load_citibike()
print(f"Citi Bike data:\n {citibike.head()}")

# plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(),
                       end=citibike.index.max(), freq='D')
# plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
# plt.plot(citibike, linewidth=1)
# plt.xlabel("Date")
# plt.ylabel("Rentals")
# plt.savefig("./hoge.png", bbox_inches="tight")

# extract the target values ( number of rentals )
y = citibike.values
# convert the time to POXIS time using "%s"
X = citibike.index.strftime("%s").astype("int").values.reshape(-1, 1)

# use the first 184 data points for training, and the rest for testing
n_train = 184

# function to evaluate and plot a regressor on a given feature set


def eval_on_features(features, target, regressor):
    # split the given features into a training and a test set
    X_train, X_test = features[:n_train], features[n_train:]
    # also split the target array
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print(f"Test-set R^2: {regressor.score(X_test, y_test):.2f}")
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.xticks(range(0, len(X)), xticks.strftime(
        "%a %m-%d"), rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred,
             '--', label="prediction test")

    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.savefig("./hoge.png", bbox_inches='tight')


# regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor = Ridge()
X_hour_week = np.hstack(
    [citibike.index.dayofweek.values.reshape(-1, 1), citibike.index.hour.values.reshape(-1, 1)])
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

poly_transformer = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)

eval_on_features(X_hour_week_onehot_poly, y, regressor)

hour = ["%02d:00" % i for i in range(0, 24, 3)]
day = ["Mon", "Tue", "Wed", "Thu", "Fir", "Sat", "Sun"]
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[regressor.coef_ != 0]
coef_nonzero = regressor.coef_[regressor.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature magnitude")
plt.ylabel("Feature")
plt.savefig("./hogehoge.png", bbox_inches='tight')
