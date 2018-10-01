from __future__ import print_function
# count time,for every step cost time
from time import time
# print evolve
import logging
# print people face after distinguish
# 将识别出来的人脸绘制出来
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
# 用来下载数据集，如果下载下来，就从本地进行提取，lfw_people是类似字典结构的对象
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
# n_samples多少个图，高，宽；
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
# 建立一个矩阵，X是关于提取特征向量的，利用data属性，可以得到；每一行是一个实例，每一类是一个特征值
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
# 提取对应的y,目标分类标记
y = lfw_people.target
# return those class one'name
# 利用target_nameas返回这个类别里面所有人的名字
target_names = lfw_people.target_names
# ensure 多少人需要区分，有多少个人
n_classes = target_names.shape[0]

print("Total dataset size:")
# 实例的个数
print("n_samples: %d" % n_samples)
# 特征向量的维度
print("n_features: %d" % n_features)
# 总共有多少类
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
# X_train是一个矩阵，
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# 组成元素的数量
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
# 为了打印每一步花费的时间
t0 = time()
# 将高维数据降维，这里随机降维
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))
# 人脸照片上提供一些特征值
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
# 记录当前时间
t0 = time()
# 训练集所有的特征向量，通过pca转化为低维的特征向量
X_train_pca = pca.transform(X_train)
# 测试集也降维
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
# 记录当前时间
t0 = time()
# C：float optional(default 1.0)
# gamma:int optional(default =3)
# 5*6 = 30 种组合
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# 建立分类器模型
# 因为是对图像进行处理，kernal选择redio biass function
# class_weight选择权重
# param_grid二维的一个格子状结构
clf = GridSearchCV(SVC(kernel='rbf', class_weight=None), param_grid)
# 超平面预测它属于哪一类
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")

print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
# 对于新来的数据集进行预测，并测试出它属于哪一类
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
# 把需要测试的测试集，测试集当中真实的y的标签，和预测的y的标签，进行比较，并填入真实人的姓名
print(classification_report(y_test, y_pred, target_names=target_names))
# 上面函数完成之后，我们可以建立n*n的方格，矩阵对角线上对应的值越多，预测的准确率越高
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib
# 可视化的打印出
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    # 建立几排，几列，进行布局
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
# 预测的函数归类标签，实际函数归类标签，以及对应的名字，
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
# 将预测出来的人，存到一个变量当中
prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
# 测试集上的特征向量矩阵，预测的title，将图像打印出来
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces
# 对 eigenface起一个名字
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# 将eigenface打印出来，分两部分，
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()