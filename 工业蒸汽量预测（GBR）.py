import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import seaborn as sns

mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

path_train = 'E:\python\工业蒸汽量预测\zhengqi_train.txt'#数据地址
path_test = 'E:\python\工业蒸汽量预测\zhengqi_test.txt'#数据地址

data = pd.read_csv(path_train, sep='\t', engine='python')
print(data.head())
co = data.corr()
print(co['target'].sort_values(ascending=False))
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(co, annot=True, vmax=1, square=True, cmap="Blues")
plt.savefig('./BluesStateRelation.png')
plt.show()

X = data.loc[:,['V0','V1','V3','V4','V7','V8','V10','V12','V15','V16','V18','V25','V26','V28','V29','V30','V31','V32','V33','V34']]
print(X.head())
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))
X_train1, X_test1, Y_train1, Y_test1 = X_train, X_test, Y_train, Y_test

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

pca = PCA(n_components=0.99)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_test.shape)

gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=150, max_depth=5, min_samples_split=6)
gbr.fit(X_train, Y_train)
Y_predict = gbr.predict(X_test)

print('score:', gbr.score(X_test, Y_test))
print("MSE:", metrics.mean_squared_error(Y_test, Y_predict))
#
# pipe = Pipeline([('ss',StandardScaler()),
#                  ('pca',PCA()),
#                  ('gbr',GradientBoostingRegressor())])
# para = {'pca__n_components':[0.8,0.9,0.95,0.98,0.99],
#         'gbr__learning_rate':[0.1,0.3],
#         'gbr__min_samples_split':[2,4,6],
#         'gbr__max_depth':[3,5,7],
#         'gbr__n_estimators':[150,200,300]}
#
# gscv = GridSearchCV(pipe,param_grid=para,cv=3)
# gscv.fit(X_train1,Y_train1)
# print("最优参数列表:", gscv.best_params_)
# print("score值：",gscv.best_score_)
# print("最优模型:", end='')
# print(gscv.best_estimator_)

test_data = pd.read_csv(path_test, sep='\t', engine='python')
X = test_data.loc[:,['V0','V1','V3','V4','V7','V8','V10','V12','V15','V16','V18','V25','V26','V28','V29','V30','V31','V32','V33','V34']]
X = ss.transform(X)
X = pca.transform(X)
y_predict = gbr.predict(X) #预测结果
print(y_predict.max())
print(y_predict.min())
np.savetxt('E:\python\工业蒸汽量预测\zhengqi_prediction.txt', y_predict)