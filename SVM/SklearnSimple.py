from sklearn import svm

# init 3 points, for that picture demo(1,1),(2,3),(2,0)
x = [[2, 0], [1, 1],[2, 3]]
# y is classLabel,which is called 分类标记，because the pre_point is below of this line,
# so it is 0,another is above of this line,equal of 1
y = [0, 0, 1]
# to struct this module,SVC is a function ,kernel is 核函数，linear is 线性
clf = svm.SVC(kernel='linear')
# x is  matrix,here is it 3 instance; y is corresponding of x,just a classlabel
clf.fit(x,y)

print(clf)
# get support vectors
print(clf.support_vectors_)
# get indices of support vectors,from 0 to begin
print(clf.support_)
# get number of support vectors for each class
print(clf.n_support_)
# to predict (2, 0)
print(clf.predict([2, 0]))