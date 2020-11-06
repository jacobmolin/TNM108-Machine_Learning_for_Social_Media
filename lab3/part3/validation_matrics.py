import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

boston = load_boston()
X=boston.data
Y=boston.target
# cv = 10
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.4, random_state=0)

print('\nlinear regression')
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nridge regression')
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlasso regression')
lasso = Lasso(alpha=0.1)
scores = cross_val_score(lasso, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\ndecision tree regression')
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nrandom forest regression')
forest = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(forest, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nlinear support vector machine')
svm_lin = svm.SVR(epsilon=0.2,kernel='linear',C=1)
scores = cross_val_score(svm_lin, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nsupport vector machine rbf')
clf = svm.SVR(epsilon=0.2,kernel='rbf',C=1.)
scores = cross_val_score(clf, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('\nknn')
knn = KNeighborsRegressor()
scores = cross_val_score(knn, X, Y, cv=cv)
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(knn, X,Y, cv=cv)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))