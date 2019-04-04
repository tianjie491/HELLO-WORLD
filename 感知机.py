import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

'''替换sans-serif字体,解决坐标轴负数的负号显示问题'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
df = pd.DataFrame(iris.data)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [2, 3, 4]])
X = data[:, :2]
Y = data[:, -1]
Y = np.array([1 if i == 1 else -1 for i in Y])


class Model:
    def __init__(self):
        self.w = np.ones(len(data[0])-1, dtype=np.float64)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def fit(self, x_train, y_train):
        iter = 3
        for i in range(iter):
            wrong_count = -1
            for d in range(len(x_train)):
                x = x_train[d]
                y = y_train[d]
                if y * self.sign(x, self.w, self.b) < 0:
                    self.w += self.l_rate * np.dot(y, x)
                    self.b += self.l_rate * y
                    wrong_count += 1
            print('第%s次迭代，分类错误的有%s' % (i, wrong_count))
        return 'Perceptron Model!'

    def score(self):
        pass


perceptron = Model()
perceptron.fit(X, Y)

x_point = np.linspace(0, 6, 10)
y_ = -(perceptron.w[0]*x_point + perceptron.b)/perceptron.w[1]
plt.plot(x_point, y_)
plt.scatter(df[:50]['petal length'], df[:50]['petal width'], c='r', label='0')
plt.scatter(df[50:100]['petal length'], df[50:100]['petal width'], c='b', label='1')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend()
plt.show()
