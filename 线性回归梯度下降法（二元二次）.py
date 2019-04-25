from urllib import request
import os
import pandas as pd
import io
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import datetime

start = datetime.datetime.now()#赋值程序开始时间
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
if os.path.exists('线性回归梯度下降法数据'):#判断文件夹是否存在
	shutil.rmtree('线性回归梯度下降法数据')#如果存在就递归删除文件夹
	os.mkdir('线性回归梯度下降法数据')#再创建一个同名新文件夹
else:
	os.mkdir('线性回归梯度下降法数据')#如果文件夹不存在就创建文件夹
os.chdir('线性回归梯度下降法数据')#把文件夹设置为当前

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
req = request.Request(url)#发起请求
req.add_header('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0')#设置请求头
response = request.urlopen(req)#打开网址
data = response.read().decode('utf-8')#读取并解码数据
#print(data)
	
names = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
dataFile = io.StringIO(data)
pd.set_option('display.max_rows', None)#设置显示所有行
pd.set_option('display.max_columns',None)#设置显示所有列
pd.set_option('display.width',1000)#设置行宽1000
DataFrame = pd.read_csv(dataFile,delim_whitespace=True,names = names)#读取CSV文件，采用空格隔离符并添加表头
#print(type(DataFrame))
with open('data.txt','w') as f:
	f.write(str(DataFrame))#把DataFrame写入data.txt
'''
ax = plt.subplot(111, projection='3d') #创建一个三维图
ax.set_title('原始三维散点图')#设置标题
ax.set_xlabel('每加仑燃料所行英里数')#设置X轴标签
ax.set_ylabel('发动机位移')#设置Y轴标签
ax.set_zlabel('加速性能')#设置Z轴标签
ax.scatter(DataFrame["mpg"][:100],DataFrame["displacement"][:100],DataFrame["acceleration"][:100],c = 'y')#设置前100位数据并用黄色表示
ax.scatter(DataFrame["mpg"][100:200],DataFrame["displacement"][100:200],DataFrame["acceleration"][100:200],c = 'r')#设置第101位到200位数据并用红色表示
ax.scatter(DataFrame["mpg"][200:300],DataFrame["displacement"][200:300],DataFrame["acceleration"][200:300],c = 'b')#设置第201位到300位数据并用蓝色表示
plt.savefig('原始三维散点图.png',dpi = 300)#保存散点图并设置像素为300
plt.show()#显示三维散点图
'''
'''
plt.title('原始二维散点图')
plt.xlabel('每加仑燃料所行英里数')#设置X轴标签
plt.ylabel('加速性能')#设置Y轴标签
plt.scatter(DataFrame["mpg"][:100],DataFrame["acceleration"][:100],c = 'y')#设置前100位数据并用黄色表示
plt.scatter(DataFrame["mpg"][100:200],DataFrame["acceleration"][100:200],c = 'r')#设置第101位到200位数据并用红色表示
plt.scatter(DataFrame["mpg"][200:300],DataFrame["acceleration"][200:300],c = 'b')#设置第201位到300位数据并用蓝色表示
plt.savefig('原始二位散点图',dpi = 300)#保存散点图并设置像素为300
plt.show()#显示二维散点图
'''
traindata = DataFrame[["mpg","displacement","acceleration"]]#data中选取三列数据作为训练数据
traindata.insert(0,'one',1)#在训练数据第一列表头添加one并将值设置为1
traindata.insert(2,'mpg²',np.power(DataFrame[["mpg"]],2))#在训练数据添加mpg²
traindata.insert(4,'displacement²',np.power(DataFrame[["displacement"]],2))#在训练数据添加displacement²
#print(traindata)
cols = traindata.shape[1]#训练数据的列数
X = traindata.iloc[0:300,0:cols-1]#把前300个样本训练特征变量赋值给X
Y = traindata.iloc[0:300,cols-1:cols]#把前300个样本训练目标结果赋值给Y
X = np.mat(X.values)#将X的值转化为矩阵
Y = np.mat(Y.values)#将Y的值转化为矩阵

for i in range(1,5):#特征缩放
	X[:,i] = (X[:,i] - min(X[:,i])) / (max(X[:,i]) - min(X[:,i])) #对X值特征缩放
#Y[:,0] = (Y[:,0] - min(Y[:,0])) / (max(Y[:,0]) - min(Y[:,0]))#对Y值特征缩放

theta = np.mat([0,0,0,0,0])#设置theta为一个1x5的矩阵
alpha = 1#设置学习率
iters = 100000#设置循环次数
cost = []#设置代价函数的值是一个列表
theta_m = []#设置theta_m的值是一个列表

theta_n = (X.T*X).I*(X.T)*Y#正规方程法求theta的值
print('采用正规方程法求得theta的值为%s：'%theta_n.T)

for i in range(iters):#循环iters次
	function = X*theta.T#求得函数h(x)的值
	inner = np.power(function-Y,2)
	cost.append((np.sum(inner))/(len(X)*2))#求得代价函数的值并添加到cost中
	theta_m.append(theta)#把theta的值并添加到theta_m中
	theta = theta - (alpha/len(X))*((X*theta.T-Y).T*X)#矩阵求梯度下降公式
print('采用梯度下降法求得theta的值为%s：'%theta)
print('最终代价函数的值为%s：'%cost[-1])

fig = plt.figure(num = 1,figsize=(9,9))
for i in range(1,10):
	Ax = plt.subplot(3,3,int(i),projection='3d') #创建一个三维图
	Ax.set_title('三维散点图及函数面')#设置标题
	Ax.set_xlabel('每加仑燃料所行英里数')#设置X轴标签
	Ax.set_ylabel('发动机位移')#设置Y轴标签
	Ax.set_zlabel('加速性能')#设置Z轴标签
	x = np.linspace(0,1,50)#对设置等差数列
	y = np.linspace(0,1,50)
	x,y = np.meshgrid(x,y)#生成坐标矩阵
	z = theta_m[i*99][0,0] + theta_m[i*99][0,1]*x + theta_m[i*99][0,2]*x**2 + theta_m[i*99][0,3]*y + theta_m[i*99][0,4]*y**2

	Ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'),label='prediction')
	Ax.scatter(X[:100,1],X[:100,3],Y[:100,0],c='y')#设置前100位数据并用黄色表示
	Ax.scatter(X[100:200,1],X[100:200,3],Y[100:200,0],c='r')#设置前100位数据并用红色表示
	Ax.scatter(X[200:,1],X[200:,3],Y[200:,0],c='b')#设置前100位数据并用蓝色表示
plt.savefig('三维散点图及函数面.png')#保存散点图并设置像素为300
plt.show()#显示三维散点图及函数面

fig = plt.figure(num = 2,figsize=(9,9))
for i in range(1,10):
	bx = plt.subplot(3,3,int(i))
	bx.plot(np.arange(i*99), cost[:i*99], 'r') #绘图
	bx.set_xlabel('迭代次数') #设置X轴标签
	bx.set_ylabel('代价函数值') #设置Y轴标签
	bx.set_title('下降曲线') #设置标题
plt.savefig('下降曲线',dpi = 300)#保存曲线图并设置像素为300
plt.show()#显示下降曲线

X = traindata.iloc[300:,0:cols-1]#把300以后的样本特征变量赋值给X
Y = traindata.iloc[300:,cols-1:cols]#把前300以后的样本训练目标赋值给Y
X = np.mat(X.values)#将X的值转化为矩阵
Y = np.mat(Y.values)#将Y的值转化为矩阵

for i in range(1,5):#特征缩放
	X[:,i] = (X[:,i] - min(X[:,i])) / (max(X[:,i]) - min(X[:,i])) #对X值特征缩放

function = X*theta.T#求得函数h(x)的值
P = []#求得的值与没添加的样本的值的错误率
for i in range(len(X)):#循环98次
	P.append((function[i]-Y[i])/Y[i])#将各个错误率赋值给P
	print('%.2f%%'%(P[i]*100))#以百分数的形式打印错误率

end = datetime.datetime.now()
print('程序运行时间：%s秒'%(end-start))#打印程序运行时间
