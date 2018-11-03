import numpy as np
import pylab as pl
from sklearn import svm

# we create 40 separable points
np.random.seed(0)    # everytime seed the same data, to ensure every result is not change
# have 20 points ,2dimension;正态分布，均值2，方差2
# 随机产生的值，一部分减去值，在左下方，一部分在右上方
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# ahead 20 points belone 0 class
Y = [0]*20 +[1]*20
print(X)
print('*'*100)
print(Y)
# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]   # 这里的w是二维的
a = -w[0]/w[1]      # the gradient of this line
xx = np.linspace(-5, 5)     # from -5 to 5 get some data
# clf.intercept_[0] is equel of w_3
yy = a*xx - (clf.intercept_[0])/w[1]        # by means of x to get y values

# plot the parallels to the separating hyperplane that pass through the support vectors
# the first lines
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
# the second lines
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

print ("w: ", w)
print ("a: ", a)

# print "xx: ", xx
# print "yy: ", yy
print ("support_vectors_: ", clf.support_vectors_)
print ("clf.coef_: ", clf.coef_)

# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()