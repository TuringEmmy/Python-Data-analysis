import numpy as np


# 定义函数，传入参数，获得b_0,b_1的值
def fitSLR(x, y):
    n = len(x)
    # 分母
    dinominator = 0
    # 分子
    numerator = 0
    for i in range(0, n):
        # SUM((x_i-x_mean)(y_i-y_mean))
        numerator += (x[i] - np.mean(x)) * (y[i] - np.mean(y))
        # SUM(x_i-x_mean)
        dinominator += (x[i] - np.mean(x)) ** 2
    print("numerator",numerator,"dinominator",dinominator)
    b1 = numerator / float(dinominator)
    b0 = np.mean(y) / float(np.mean(x))
    return b0, b1


# 获取y的估计值y=b0 + x * b1
def predict(x, b0, b1):
    return b0 + x * b1


x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
b0, b1 = fitSLR(x, y)
print("intercept:", b0, " slope:", b1)
x_test = 6
y_test = predict(6, b0, b1)
print("y_test:", y_test)
