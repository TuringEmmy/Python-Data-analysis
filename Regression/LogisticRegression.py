import numpy as np
import random

# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    """
    梯度算法
    :param x: 之前所生成的x
    :param y: 之前所生成的y
    :param theta: 要学习的参数值，theta1,theta2...
    :param alpha: 学习率
    :param m: 对应公式当中的M,生成多少个实例
    :param numIterations: 对更新法则重复大少次
    :return: 
    """
    # 将X转置
    xTrans = x.transpose()

    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        # 偏导数
        gradient = np.dot(xTrans, loss) / m
        # update
        # 更新法则
        theta = theta - alpha * gradient
    return theta


def genData(numPoints, bias, variance):
    """
    创建一些数据
    :param numPoints: 多少个实例
    :param bias: 随机生成y的时候，随机加一些偏好
    :param variance: 统计学上的一些定义，方差
    :return: 
    """
    # 声明变量
    # numPoint表示行
    x = np.zeros(shape=(numPoints, 2))
    # 行数与x保持一致
    y = np.zeros(shape=numPoints)
    # basically a straight line
    # 警告，不包末尾位值
    for i in range(0, numPoints):
        # bias feature
        # 每一行第一列
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        # 每一行，加入偏差再加上随机在（0,1）之间上的数
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
print("x---------------------------------",x)
print("y---------------------------------",y)
# 打印x的形状
m, n = np.shape(x)
# 打印y的形状
m_y = np.shape(y)
numIterations= 1000000
# 拟合步长
alpha = 0.0005
# n维数据
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)