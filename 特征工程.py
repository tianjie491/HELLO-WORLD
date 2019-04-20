from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
# from minepy import MINE

# 导入IRIS数据集
iris = load_iris()

# 标准化
SS = StandardScaler()
SSdata = SS.fit_transform(iris.data)

# 区间缩放
MM = MinMaxScaler()
MMdata = MM.fit_transform(iris.data)

# 归一化
Nm = Normalizer()
Nmdata = Nm.fit_transform(iris.data)

# 二值化
Bin = Binarizer(threshold=2)
Bindata = Bin.fit_transform(iris.data)

# 哑编码
OHE = OneHotEncoder()
OHEtarget = OHE.fit_transform(iris.target.reshape(-1, 1))

# 缺失值计算，返回值为计算缺失值后的数据
# 参数missing_value为缺失值的表示形式，默认为NaN
# 参数strategy为缺失值填充方式，默认为mean（均值）
Imp = Imputer()
Impdata = Imp.fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))

# 多项式变换
Poly = PolynomialFeatures()
Polydata = Poly.fit_transform(iris.data)

# 方差选择法
VT = VarianceThreshold(threshold=0.5)
VTdata = VT.fit_transform(iris.data)

# 相关系数法
# pearsonrdata = SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

# 卡方检验
chidata = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)


# 互信息法
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
#
#
# micdata = SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)


# 特征矩阵
print(micdata)

# 目标向量
# print(OHEtarget)
