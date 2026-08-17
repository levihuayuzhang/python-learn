from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

from sklearn import datasets

loaded_data = fetch_california_housing()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])

X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)

import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.show()
