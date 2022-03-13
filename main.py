
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from node import Node


data = datasets.load_diabetes()
X = pd.DataFrame(data.data)
X.columns = data.feature_names
y = pd.DataFrame(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


node = Node(X_train, y_train, X_val = X_test, y_val = y_test, depth = 0, max_depth = 10)
# node = Node(X_train, y_train, X_val = None, y_val = None, depth = 0, max_depth = 2)
node.grow_tree()
max_height = None
y_train_pred = node.predict(X_train, max_height=max_height)
y_test_pred = node.predict(X_test, max_height=max_height)
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)


from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X_train,y_train)
y_train_pred = regr_1.predict(X_train)
y_test_pred = regr_1.predict(X_test)
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)











