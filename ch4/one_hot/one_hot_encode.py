import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../mglearn/data/adult.data', header=None,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

data = data[['age', 'workclass', 'education', 'gender',
             'hours-per-week', 'occupation', 'income']]

# check categorical data
# for col in data.columns:
#     if data[col].dtype == 'O':
#         pprint(data[col].value_counts())
#         print()

# print(f"original features:\n {list(data.columns)}\n")
data_dummies = pd.get_dummies(data)
# print(f"Features after get_dummies:\n {list(data_dummies.columns)}")

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
# print(f"X.shape: {X.shape}, y.shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(f"Test score: {logreg.score(X_test, y_test):.2f}")
