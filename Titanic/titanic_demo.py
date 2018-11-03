import pandas as pd
import matplotlib.pyplot as plt


# 首先用Spyder载入了泰坦尼克号的CSV数据文件，并打印了一下列名与样本个体数
train = pd.read_csv('./titanic_train.csv')
print(train.columns.values.tolist())
print(len(train))

# 从数据上并不能看出性别年龄等特征与是否幸存的关系。现在利用透视表查看仓位等级、性别与存活率的关系
class_survived= train.pivot_table(index="Pclass",values="Survived")#仓位等级与存活率
sex_survived=train.pivot_table(index="Sex",values="Survived")#性别与存活率
print(class_survived)
print(sex_survived)
# 这里发现仓位等级越高存活率越大，并且女性的存活率要远高于男性。

# 接下来利用绘图工具matplot的柱形图简单统计了一下年龄与存活率的关系：
age = train["Age"]
less5 = train[age <= 5]
bigger5 = train[5 < age]
less15 = train[age < 15]
beyond60 = train[age > 60]

percent5 = len(less5[less5["Survived"] == 1]) / len(less5)
percent15 = len(less15[less15["Survived"] == 1]) / len(less15)
percentB5 = len(bigger5[bigger5["Survived"] == 1]) / len(bigger5)
percentbeyond60 = len(beyond60[beyond60["Survived"] == 1]) / len(beyond60)

x = ["<5", "<15", ">5", ">60"]
y = [percent5, percent15, percentB5, percentbeyond60]
plt.bar(x, y)

# 从图中可以看出，小于五岁的孩子存活率最高，15岁的次之。而大于60岁的存活率最低。
#
# 之后来看一下家庭成员数与存活率的关系：

train["FamilySize"] = train["SibSp"] + train["Parch"]  # 家庭成员数由这两个列组成，新建列FamilySize
size = train[["FamilySize", "Survived"]]
size1 = size[size["FamilySize"] == 1]
size23 = size[size["FamilySize"].apply(lambda x: 1 < x < 4) == True]
size45 = size[size["FamilySize"].apply(lambda x: 3 < x < 6) == True]
size5plus = size[size["FamilySize"] >= 5]

survivdPercent1 = len(size1[size1["Survived"] == 1]) / len(size1)  # 计算各个家庭数目及存活率
survivdPercent23 = len(size23[size23["Survived"] == 1]) / len(size23)
survivdPercent45 = len(size45[size45["Survived"] == 1]) / len(size45)
survivdPercent5plus = len(size5plus[size5plus["Survived"] == 1]) / len(size5plus)

x = ["1", "23", "45", "5 plus"]
y = [survivdPercent1, survivdPercent23, survivdPercent45, survivdPercent5plus]
plt.bar(x, y)



# 发现家庭成员数与存活率是有关系的。
# 所以从以上分析得出结论，存活率与年龄，性别，仓位等级有一定联系。接下来，就利用Python库中的随机森林模型对样本进行分析:
# 1. 要做的事情第一步是处理缺失值，以保证它对结果没有影响：

#train["Age"]=train["Age"].fillna(train["Age"].median())#填充年龄空行
train=train.dropna(subset=["Age"])#删除年龄空行
# 可以用年龄的中间值填补空行，或者直接剔除年龄空值。如果样本量非常巨大，那么少量地删除个体对结果的影响是非常小的。
# 2. 由于Python库处理数值比较方便，所以这里将男女表示为1和0
train.loc[train["Sex"]=="male","Sex"]=1#用0和1代表男女
train.loc[train["Sex"]=="female","Sex"] = 0
# 3. 声明特性列名
predictors=["Sex","Age","Pclass","FamilySize"]
# 4. 随机森林分类器
# 随机森林是由多个决策树组成，决策树是根据各个特性对结果的影响从大到小排列，逐一排查并最终确定最终结果。
# 例如在这里，对是否存活影响最大的因素为性别，那么建立的决策树的根节点就会判断此人是男人还是女人。之后再根据其他因素最终得出是否幸存。
# 但是决策树并不是分支越多越好，这样会导致叶子节点的个体个数过少，导致过拟合。所以在建立决策树时，会根据不同情况对决策树进行限制，例如限制节点深度、限制叶子节点的个数、叶子节点的最少样本数等。
# 在这里，随机森林就是由多个决策树组成。他可以自动给出哪些特性比较重要，并且删除对结果没有影响或者影响极小的特性（对特性加噪并检验对结果影响大小）。
# 此处声明一个随机森林模型：
alg=RandomForestClassifier(random_state=2,n_estimators=5,min_samples_split=3,min_samples_leaf=1)
scores=cross_validation.cross_val_score(alg, train[predictors], train["Survived"], cv=kf)#cv：代表不同的cross validation的方法
print(scores.mean())