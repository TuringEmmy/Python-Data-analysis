import numpy as np


# 定义一个函数，返回双曲函数
def tanh(x):
    return np.tanh(x)

# 导数
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

# 逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))

# 导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """  
        :param layers: A list containing the number of units in each layer.
        Should be at least two values  
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"  
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        # 权重
        self.weights = []
        # layers是list,每层的神经元，从第一层，到最后一层减1
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # X:训练集，y:class_label预测标记，learning_rate学习率，epochs每次更新一遍，采区抽样来，因为数据量太大，这里每次抽样epochs
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)  # X至少为二维的
        # 初始化矩阵，全是一，以及多少行，多少列
        # X.shape[0]行数
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            # 随机抽取一行
            i = np.random.randint(X.shape[0])
            # 取第i行
            a = [X[i]]

            for l in range(len(self.weights)):  # going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[
                    l])))  # Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  # Computer the error at the top layer
            # 算更新过的误差
            deltas = [
                error * self.activation_deriv(a[-1])]  # For output layer, Err calculation (delta is updated error)

            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                # 隐藏层的函数
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            # 将层数的顺序颠倒过来
            deltas.reverse()
            for i in range(len(self.weights)):
                # i从倒数第二层开始
                layer = np.atleast_2d(a[i])
                # deltas存的所预测更新过的误差
                delta = np.atleast_2d(deltas[i])
                # 更新权重
                self.weights[i] += learning_rate * layer.T.dot(delta)
    # 模型训练后，如何预设
    def predict(self, x):
        # 先定义成array
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            # 不需要保存前面的值，所以这里不需要append
            a = self.activation(np.dot(a, self.weights[l]))
        return a
