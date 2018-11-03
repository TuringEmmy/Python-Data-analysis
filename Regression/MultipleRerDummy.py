from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r".\DeliveryDummyDone.csv"
# dataPath = r".\Delivery.csv"
# 提取数据，然后按逗号分隔
deliveryData = genfromtxt(dataPath, delimiter=',')

print("data")
print(deliveryData)

X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print("X:")
print(X)
print("Y: ")
print(Y)

# 线性回归模型
regr = linear_model.LinearRegression()
# 对X,Y 进行建模
regr.fit(X, Y)

print("coefficients")
# coef_算出参数预测b_0,b_1,b_2
print(regr.coef_)
print("intercept: ")
print(regr.intercept_)

xPred = [[102, 6]]
yPred = regr.predict(xPred)
print("predicted y: ")
print(yPred)
