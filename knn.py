import numpy as np

#定义欧式
def euclidean_distance(x1,y1):
    distance = np.sqrt(np.sum((x1-y1)**2))
    return distance


def knn(X_train, X_test, y_train, k):
    y_pred = []
    for x_test in X_test:    #遍历测试集中每个点与训练集的每个点的距离
        distances = [euclidean_distance(x_train, x_test) for x_train in X_train]
        index = np.argsort(distances)[:k]  #将distances中的值从小到大以索引排序，取出前k个
        labels =[y_train[i] for i in index]  #将索引转为值存到列表
        set_labels = list(set(labels))  #按标签排序
        myset = set(set_labels)
        counts = []
        for i in myset:
            count = set_labels.count(i)
            counts.append(count)
        f_label = max(counts)
        y_pred.append(f_label)
    return np.array(y_pred)


#test

# 设置随机数种子
np.random.seed(2021)

# 生成100个随机坐标
n = 100
X = np.random.randint(-100, 100, size=(100, 2))
# 随机赋予每个数据点一个类别标签（0或1）
y = np.random.randint(0, 5, n)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

result = knn(X_train, X_test, y_train, 5)
print(result)