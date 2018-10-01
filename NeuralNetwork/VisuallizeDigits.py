from sklearn.datasets import load_digits


digits = load_digits()
print(digits.data.shape)
import pylab as pl
pl.gray()
for i in range(10):
    pl.matshow(digits.images[i])
    pl.show()