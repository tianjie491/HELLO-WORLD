import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

# Sequential模型结构构造，Dense全连接层（单元数，激活函数，第一行要有输入特征数量）
model = keras.Sequential([
    layers.Dense(32, activation='sigmoid', input_shape=(13,), name='h1'),
    layers.Dense(32, activation='sigmoid', name='h2'),
    layers.Dense(32, activation='sigmoid', name='h3'),
    layers.Dense(1, name='h4')
])

# compile配置模型，optimizers优化器（这里使用SGD随机梯度下降），loss损失函数，metrics度量方式
model.compile(optimizer=keras.optimizers.SGD(0.3),
              loss='mean_squared_error',
              metrics=['mse'])

# 模型信息摘要
model.summary()

# 训练模型
model.fit(x_train, y_train, batch_size=363, epochs=1000, validation_split=0.1, verbose=1)

# 评估模型
result = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(result)
