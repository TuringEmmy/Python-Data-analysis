from sklearn import neighbors
from sklearn import datasets

# 调用KNN的分离器
knn = neighbors.KNeighborsClassifier()
# 赋值变量，datasets里面有数据集，返回一个数据库
iris = datasets.load_iris()
# 150行每个画的特征向量
print(iris)
# 模型的建立
knn.fit(iris.data, iris.target)
# 这里的KNN启动的是默认的值
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print (predictedLabel)