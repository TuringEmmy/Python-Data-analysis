import numpy as np
from astropy.units import Ybarn
import math


def computeCorrelation(X, Y):
    """
    计算相关度
    :param X: 
    :param Y: 
    :return: 
    """
    # EX
    xBar = np.mean(X)
    # EY
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    # 使用相关度公式
    SST = math.sqrt(varX * varY)
    return SSR / SST
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)
    # Polynomial Coefficients
    # 转换成list
    results['polynimial'] = coeffs.tolist()
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    # 预测值与平均值的差的平方
    ssreg = np.sum((yhat - ybar)**2)
    # 打印截距
    print("打印截距ssreg:",str(ssreg))
    # 求平方和
    # 打印斜距
    sstot = np.sum((y - ybar)**2)
    print("打印斜距sstot:",str(sstot))
    results['determination'] = ssreg / sstot
    return results

testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print("r:",computeCorrelation(testX, testY))
print("r^2:",str(computeCorrelation(testX,testY)**2))


print(polyfit(testX, testY,1)['determination'])
print(polyfit(testX, testY,1))