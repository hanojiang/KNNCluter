from sklearn import neighbors, datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()

print(iris)

knn.fit(iris.data,iris.target)
knn_predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(knn_predict)
