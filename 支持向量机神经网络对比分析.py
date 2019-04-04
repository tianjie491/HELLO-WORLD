from urllib import request
import os
import pandas as pd
import io
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import datetime
import copy

def get_data():#得到数据并可视化
    start = datetime.datetime.now()#赋值程序开始时间
    if os.path.exists('SVM与BP算法对比分析数据'):#判断文件夹是否存在
        shutil.rmtree('SVM与BP算法对比分析数据')#如果存在就递归删除文件夹
        os.mkdir('SVM与BP算法对比分析数据')#再创建一个同名新文件夹
    else:
        os.mkdir('SVM与BP算法对比分析数据')#如果文件夹不存在就创建文件夹
    os.chdir('SVM与BP算法对比分析数据')#把文件夹设置为当前
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    req = request.Request(url)#发起请求
    response = request.urlopen(req)#打开网址
    data = response.read().decode('utf-8')#读取并解码数据
    names = ['sepal_length','sepal_width','petal_length','petal_width','class']
    dataFile = io.StringIO(data)
    pd.set_option('display.max_rows', None)#设置显示所有行
    pd.set_option('display.max_columns',None)#设置显示所有列
    pd.set_option('display.width',1000)#设置行宽1000
    DataFrame = pd.read_csv(dataFile,names = names)#读取CSV文件，采用空格隔离符并添加表头
    with open('data.txt','w') as f:
        f.write(str(DataFrame))#把DataFrame写入data.txt
    ax = plt.subplot(111, projection='3d') #创建一个三维图
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    ax.set_title('鸢尾花三维散点图')#设置标题
    ax.set_xlabel('花萼长度')#设置X轴标签
    ax.set_ylabel('花萼宽度')#设置Y轴标签
    ax.set_zlabel('花瓣长度')#设置Z轴标签
    ax.scatter(DataFrame['sepal_length'][:50],DataFrame['sepal_width'][:50],DataFrame['petal_length'][:50],c = 'y')#设置前50位数据并用黄色表示
    ax.scatter(DataFrame['sepal_length'][50:100],DataFrame['sepal_width'][50:100],DataFrame['petal_length'][50:100],c = 'r')#设置第50位到100位数据并用红色表示
    ax.scatter(DataFrame['sepal_length'][100:150],DataFrame['sepal_width'][100:150],DataFrame['petal_length'][100:150],c = 'b')#设置第100位到150位数据并用蓝色表示
    plt.savefig('鸢尾花三维散点图.png',dpi = 300)#保存散点图并设置像素为300
    #plt.show()#显示三维散点图
    return DataFrame
    
def BP_data():#处理数据
    data = get_data()
    np.random.seed(2)#设置伪随机数种子为0
    num  = np.arange(len(data))
    np.random.shuffle(num)
    X = data[['sepal_length','sepal_width','petal_length','petal_width']]
    X = X.values#将X转化为数组
    Y = data['class']
    y = []
    for i in range(len(Y)):#将Y值转化为1,0,0   0,1,0   0,0,1
        if Y[i] == 'Iris-setosa':
            y.append([1,0,0])
        elif Y[i] =='Iris-versicolor':
            y.append([0,1,0])
        else:
            y.append([0,0,1])
    new_X = []
    new_y = []
    for i in num:
        new_X.append(X[i])
        new_y.append(y[i])
    X = np.mat(new_X)
    X_max = np.max(X)
    X_min = np.min(X)
    X = (X-X_min)/(X_max-X_min)#归一化
    y = np.mat(new_y)
    train_X = X[:90]#选取前90位为训练集
    train_y = y[:90]#选取前90位为训练集
    cv_X = X[90:120]#选取中间30位为交叉测试集
    cv_y = y[90:120]#选取中间30位为交叉测试集
    test_X = X[120:]#选取中间30位为交叉测试集
    test_y = y[120:]#选取后30位为测试集
    return train_X,train_y,cv_X,cv_y,test_X,test_y

def parameter(n_hide = 8):#初始化参数
    np.random.seed(0)#设置伪随机数种子为0
    W1 = np.random.rand(4,n_hide)*(2*0.01)-0.01#W1随机初始化
    b1 = np.zeros((1, n_hide))
    W2 = np.random.rand(n_hide,3)*(2*0.01)-0.01#W2随机初始化
    b2 = np.zeros((1, 3))
    parameter = {}
    parameter = {'W2':W2,'b2':b2,'W1':W1,'b1':b1}
    return parameter

def build_model(X,y,parameters,iters = 50000,alpha = 0.01,Lambda = 0.01):#建立模型，输出参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    cost = []#设置代价函数的值是一个列表
    gradients = {}#设置梯度为一个字典
    parameters_new = {}#设置参数为一个字典
    for i in range(iters):
        z1 = np.dot(X,W1) + b1# 输入层向隐藏层正向传播
        a1 = 1/(1+np.exp(-z1))#隐藏层激活函数
        z2 = np.dot(a1,W2) + b2# 隐藏层向输出层正向传播
        a2 = 1/(1+np.exp(-z2))#输出层激活函数
        cost.append(-(1/len(X))*np.sum(np.multiply(np.log(a2),y)+np.multiply(np.log(1-a2),1-y))+(Lambda/(2*len(X)))*(np.sum(W1*W1)+np.sum(W2*W2)))#求出代价函数
        delta2 = a2-y
        dW2 = (1/len(X))*np.dot(a1.T,delta2)+(Lambda/(len(X)))*W2#损失函数对w2的偏导数
        db2 = (1/len(X))*np.sum(delta2, axis=0)#损失函数对b2的偏导数
        delta1 = np.multiply(np.dot(delta2,W2.T) ,np.multiply(a1,(1-a1)))#损失函数对z1的偏导数
        dW1 = (1/len(X))*np.dot(X.T, delta1)+(Lambda/(len(X)))*W1#损失函数对w1的偏导数
        db1 = (1/len(X))*np.sum(delta1, axis=0)#损失函数对b1的偏导数
        W1 += -alpha * dW1
        b1 += -alpha * db1
        W2 += -alpha * dW2
        b2 += -alpha * db2
    gradients = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1,}
    parameters_new = {'W2':W2,'b2':b2,'W1':W1,'b1':b1}
    return parameters_new,cost,gradients
    
def draw_cost(iters = 500000,alpha = 0.01,Lambda = 0.01):
    start = datetime.datetime.now()#赋值程序开始时间
    parameters_new,cost,gradients = build_model(train_X,train_y,parameters,iters,alpha,Lambda)
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    fig, bx = plt.subplots()#创建一个画布
    bx.plot(np.arange(iters), cost, 'r') #训练集代价函数随迭代次数变化
    bx.set_xlabel('迭代次数') #设置X轴标签
    bx.set_ylabel('代价函数值') #设置Y轴标签
    bx.set_title('训练集代价函数随迭代次数变化曲线') #设置标题
    plt.savefig('训练集代价函数随迭代次数变化曲线',dpi = 300)#保存曲线图并设置像素为300
    plt.show()#显示下降曲线
    print('迭代次数为%s时最终的代价函数值为：%s'%(iters,cost[-1]))
    end = datetime.datetime.now()
    print('迭代次数为%s时程序运行时间：%s秒'%(iters,end-start))#打印程序运行时间
    return cost[-1]

def gradient_check(parameters,epsilon=1e-7):#梯度检验
    gradient = []
    for key in parameters.keys():#遍历参数的值
        i = parameters[key]
        shape0 = np.shape(i)[0]#参数的行数
        shape1 = np.shape(i)[1]#参数的列数
        values = i.reshape(shape0*shape1,1)#参数转化为向量
        for j in range(len(values)):#遍历向量
            parameters_plus = copy.deepcopy(parameters)# 此处必须用深拷贝
            values_plus = np.copy(values)#浅拷贝
            values_plus[j][0] = values_plus[j][0] + epsilon#加上一个数
            values_plus = values_plus.reshape(shape0,shape1)#还原参数形状
            parameters_plus[key] = values_plus#将加大的值赋值给参数
            parameters_new,cost_plus,gradients = build_model(train_X,train_y,parameters_plus,iters = 1,alpha = 0.01)#不循环
            
            parameters_min = copy.deepcopy(parameters)# 此处必须用深拷贝
            values_min = np.copy(values)#浅拷贝
            values_min[j][0] = values_min[j][0] - epsilon#减去一个数
            values_min = values_min.reshape(shape0,shape1)#还原参数形状
            parameters_min[key] = values_min#将加大的值赋值给参数
            parameters_new,cost_min,gradients = build_model(train_X,train_y,parameters_min,iters = 1,alpha = 0.01)#不循环
            gradient.append((cost_plus[0]-cost_min[0])/(2 * epsilon))#求出梯度值
    parameters_new,cost,gradients = build_model(train_X,train_y,parameters,iters = 1,alpha = 0.01)#不循环
    gradients_values = []
    for key in gradients.keys():#遍历梯度的值
        i = gradients[key]
        shape0 = np.shape(i)[0]#参数的行数
        shape1 = np.shape(i)[1]#参数的列数
        values = i.reshape(shape0*shape1,1)#参数转化为向量
        for j in range(len(values)):#遍历向量
            gradients_values.append(float(values[j]))
    gradient= np.mat(gradient)
    gradients_values =np.mat(gradients_values)
    diff = np.linalg.norm(gradient - gradients_values)/(np.linalg.norm(gradient) + np.linalg.norm(gradients_values))
    print('梯度检验的差值为：%s'%diff)

def iters_choose(iters = 50000,alpha = 0.1):
    parameters_new,cost_train,gradients = build_model(train_X,train_y,parameters,iters = iters,alpha = alpha,Lambda = 0.01)
    print('迭代%s次训练集代价函数的最小值为：%s'%(iters,cost_train[-1]))
    cost_cv_min = []
    for i in range(1,iters+1,1000):#遍历iters/1000次，左闭右开区间
        np.random.seed(0)#parameters为nocal值，此处不能申明nonnocal，只能重新运算参数初始化！
        W1 = np.random.rand(4,8)*(2*0.01)-0.01
        b1 = np.zeros((1, 8))
        W2 = np.random.rand(8,3)*(2*0.01)-0.01
        b2 = np.zeros((1, 3))
        parameter = {}
        parameter = {'W2':W2,'b2':b2,'W1':W1,'b1':b1}
        parameters_new,cost,gradients = build_model(train_X,train_y,parameters = parameter,iters = i,alpha = alpha,Lambda = 0.01)#求出迭代1,2,3。。。次的新参数
        parameter,cost_cv,gradients = build_model(cv_X,cv_y,parameters = parameters_new,iters = 1,alpha = alpha,Lambda = 0.01)#将新参数带入模型，求出交叉测试集下的代价函数值
        cost_cv_min.append(cost_cv[-1])
        print('迭代%s次交叉测试集代价函数的值为：%s'%(i-1,cost_cv[-1]))
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    fig, bx = plt.subplots()
    bx.plot(np.arange(iters), cost_train, 'r') #训练集代价函数随迭代次数变化
    bx.plot(np.arange(1,iters+1,1000), cost_cv_min, 'b') #交叉测试集代价函数随迭代次数变化
    bx.set_xlabel('迭代次数') #设置X轴标签
    bx.set_ylabel('代价函数值') #设置Y轴标签
    bx.set_title('代价函数随迭代次数变化曲线') #设置标题
    plt.savefig('代价函数随迭代次数变化曲线',dpi = 300)#保存曲线图并设置像素为300
    plt.show()#显示下降曲线

def Lambda_choose():
    lambda_values = np.arange(0,0.1,0.01)
    cost_min_train = []
    cost_min_cv = []
    for i in lambda_values:
        np.random.seed(0)#parameters为nocal值，此处不能申明nonnocal，只能重新运算参数初始化！
        W1 = np.random.rand(4,8)*(2*0.01)-0.01
        b1 = np.zeros((1, 8))
        W2 = np.random.rand(8,3)*(2*0.01)-0.01
        b2 = np.zeros((1, 3))
        parameter = {}
        parameter = {'W2':W2,'b2':b2,'W1':W1,'b1':b1}
        parameters_new,cost,gradients = build_model(train_X,train_y,parameters,iters = 50000,alpha = 0.1,Lambda = i)
        cost_min_train.append(cost[-1])
        print('正则化参数为%s时训练集代价函数最小值为：%s'%(i,cost[-1]))
        parameters_new,cost,gradients = build_model(cv_X,cv_y,parameters = parameters_new,iters = 1,alpha = 0.1,Lambda = i)
        cost_min_cv.append(cost[-1])
        print('正则化参数为%s时交叉测试集代价函数值为：%s'%(i,cost[-1]))

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    fig, bx = plt.subplots()#创建一个画布
    bx.plot(lambda_values, cost_min_train, 'r') #训练集代价函数最小值随正则化参数变化
    bx.plot(lambda_values, cost_min_cv, 'b') #测试集代价函数最小值随正则化参数变化
    bx.set_xlabel('正则化参数') #设置X轴标签
    bx.set_ylabel('代价函数值') #设置Y轴标签
    bx.set_title('代价函数随正则化参数变化曲线') #设置标题
    plt.savefig('代价函数随正则化参数变化曲线',dpi = 300)#保存曲线图并设置像素为300
    plt.show()#显示下降曲线
    
def  n_hide_choose():
    n_hide_values = np.arange(4,21)
    cost_min_train = []
    cost_min_cv = []
    for i in n_hide_values:
        start = datetime.datetime.now()#赋值程序开始时间
        np.random.seed(0)#parameters为nocal值，此处不能申明nonnocal，只能重新运算参数初始化！
        W1 = np.random.rand(4,i)*(2*0.01)-0.01
        b1 = np.zeros((1, i))
        W2 = np.random.rand(i,3)*(2*0.01)-0.01
        b2 = np.zeros((1, 3))
        parameter = {}
        parameter = {'W2':W2,'b2':b2,'W1':W1,'b1':b1}
        parameters_new,cost,gradients = build_model(train_X,train_y,parameters,iters = 50000,alpha = 0.1,Lambda = 0.02)
        cost_min_train.append(cost[-1])
        print('隐藏层神经元个数为%s时训练集代价函数最小值为：%s'%(i,cost[-1]))
        parameters_new,cost,gradients = build_model(cv_X,cv_y,parameters = parameters_new,iters = 1,alpha = 0.1,Lambda = 0.02)
        cost_min_cv.append(cost[-1])
        print('隐藏层神经元个数为%s时交叉测试集代价函数值为：%s'%(i,cost[-1]))
        end = datetime.datetime.now()
        print('隐藏层神经元个数为%s时程序运行时间：%s秒'%(i,end-start))#打印程序运行时间

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    fig, bx = plt.subplots()#创建一个画布
    bx.plot(n_hide_values, cost_min_train, 'r') #训练集代价函数最小值随隐藏层神经元个数变化
    bx.plot(n_hide_values, cost_min_cv, 'b') #测试集代价函数最小值随隐藏层神经元个数变化
    bx.set_xlabel('隐藏层神经元个数') #设置X轴标签
    bx.set_ylabel('代价函数值') #设置Y轴标签
    bx.set_title('代价函数随隐藏层神经元个数变化曲线') #设置标题
    plt.savefig('代价函数随隐藏层神经元个数变化曲线',dpi = 300)#保存曲线图并设置像素为300
    plt.show()#显示下降曲线

def accuracy(X,y):
    parameters_new,cost,gradients = build_model(train_X,train_y,parameters,iters = 50000,alpha = 0.1,Lambda = 0.02)
    W1 = parameters_new['W1']
    b1 = parameters_new['b1']
    W2 = parameters_new['W2']
    b2 = parameters_new['b2']
    z1 = np.dot(X,W1) + b1# 输入层向隐藏层正向传播
    a1 = 1/(1+np.exp(-z1))#隐藏层激活函数
    z2 = np.dot(a1,W2) + b2# 隐藏层向输出层正向传播
    a2 = 1/(1+np.exp(-z2))#输出层激活函数

    temp = 0
    for i in range(len(y)):
        if np.argmax(y[i]) == np.argmax(a2[i]):
            temp += 1
    acc = temp/len(y)
    print('对测试集进行测试，得出准确率为：%s%%'%(acc*100))

if __name__ == "__main__":
    train_X,train_y,cv_X,cv_y,test_X,test_y = BP_data()
    parameters = parameter()
    cost_min = draw_cost()#由于此处运算过程中会迭代更新初始化参数，所以之后进行梯度检验等步骤需要重新初始化参数
    parameters = parameter()
    gradient_check(parameters)
    parameters = parameter()
    iters_choose(iters = 50000,alpha = 0.1)
    parameters = parameter()
    Lambda_choose()
    parameters = parameter()
    n_hide_choose()
    parameters = parameter(n_hide = 20)
    accuracy(test_X,test_y)