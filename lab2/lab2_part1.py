import matplotlib.pyplot as plt 
import seaborn as sns; sns.set() 
import numpy as np

rng = np.random.RandomState(1) 
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50) 
# plt.scatter(x, y)
# plt.show()

from sklearn.linear_model import LinearRegression 
model = LinearRegression(fit_intercept=True) 
model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y) 
plt.plot(xfit, yfit) 
print("Model slope: ", model.coef_[0]) 
print("Model intercept:", model.intercept_)
# plt.show()

rng = np.random.RandomState(1)
X = 10 * rng.rand(100, 3)
y = 0.5 + np.dot(X, [1.5, -2., 1.]) 
model.fit(X, y) 
print(model.intercept_)
print(model.coef_)
