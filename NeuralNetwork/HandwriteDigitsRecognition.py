import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
# 分成k份，进行交叉验算，k-1份作为训练集，剩余一份当做测试集
from sklearn.cross_validation import train_test_split

# 装载数据
digits = load_digits()
# 将数据加载到X上面
X = digits.data
y = digits.target
# 将所有值转换到0~1之间
X -= X.min() # normalize the values to bring them into the range 0-1
X /= X.max()

# 每个图8*8,固输入层有64个维度（和特征向量是一样的），
# 隐藏层100
# 输出层10个（因为区分10个数字）
nn = NeuralNetwork([64,100,10],'logistic')
# 将原始数据分成两部分，
X_train, X_test, y_train, y_test = train_test_split(X, y)
# LabelBinarizer
# 将数据转化成0,1的类型
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print ("start fitting")
nn.fit(X_train,labels_train,epochs=3000)
# 创建空的list
predictions = []
# 对每一行都进行predict
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i] )
    # 看第几个数对应最大的概率
    predictions.append(np.argmax(o))
# 可以看出最终有多少个是正确的
print (confusion_matrix(y_test,predictions))
# 算精确度
print (classification_report(y_test,predictions))