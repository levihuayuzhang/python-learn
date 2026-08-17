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

print(model.coef_)
print(model.intercept_)

print(model.get_params())
print(model.score(data_X, data_y))
