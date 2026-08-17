from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# # y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))

# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
# print(scores.mean())
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    # k_scores.append(scores.mean())
    loss = -cross_val_score(knn, X, y, cv=10, scoring="neg_mean_squared_error")
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()
