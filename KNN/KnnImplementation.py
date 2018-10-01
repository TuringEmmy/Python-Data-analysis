# Example of kNN implemented from Scratch in Python
import csv
import random
import math
import operator


# 将使用的数据集装载进来
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    '''
    :param filename: 文件的名称所在的路径
    :param split: 数据集，分为两部分，一部分训练集，一部分测试集，定好的界限
    :param trainingSet: 训练集
    :param testSet: 测试集
    '''
    with open(filename, 'rt') as csvfile:
        # 读取所有的行
        lines = csv.reader(csvfile)
        # 转化成list的形式
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                # 加入训练集
                trainingSet.append(dataset[x])
            else:
                # 加入测试集
                testSet.append(dataset[x])


# 计算两点之间距离的开方
def euclideanDistance(instance1, instance2, length):
    '''
    :param instance1: 两个实例，可能是多维的
    :param instance2: 
    :param length: 表示维度
    :return: 多维点距离开方数
    '''
    distance = 0
    # 对每一维激进型差值的运算
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# 返回最近的k个邻居
def getNeighbors(trainingSet, testInstance, k):
    '''
    :param trainingSet: 训练集
    :param testInstance: 测试集中的一个实例
    :param k: 训练集当中选出k个值
    :return: 
    '''
    # 创建一个容器
    distances = []
    # 测试实例纬度减1
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        # 将每次算出的结果加进去
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    # 获取最近的k个邻居
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# 根据邻居，利用少数服从多数的原则
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # 将每一类投票进行排序
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]  # 因为拍过需，第一个返回即可，最大值


# 测算实际值与测算值之间的差距，求准确率
def getAccuracy(testSet, predictions):
    '''
    :param testSet: 测试集
    :param predictions: 预测
    '''
    # 初始壶化变量
    correct = 0
    for x in range(len(testSet)):
        # -1表示最后一个值，对应他的Label,如果想等，证明预测对了一个
        if testSet[x][-1] == predictions[x]:
            correct += 1
    # 转换成百分比
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    # 分两步部分标准，0~1的分布当中，即load数据的时候，将2/3的数据互粉为训练集，其余划分为测试集
    split = 0.67
    # 装载数据
    loadDataset(r'data.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    # 去最近的3个邻居
    k = 3
    for x in range(len(testSet)):
        # 找到最近的邻居
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        # 取到它归为哪一类
        result = getResponse(neighbors)
        # 将其加到所有predictions里面
        predictions.append(result)
        # 打印每个变量预测出来的归类是多少
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    # 求出到底有多少预测对了
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
