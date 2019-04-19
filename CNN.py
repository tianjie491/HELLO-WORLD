from tensorflow.python.keras import layers
from tensorflow.python import keras
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/python/Lib/site-packages/tensorflow/python/keras/datasets/mnist.npz'
f = np.load(path)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

plt.imshow(x_train[0])
plt.show()
print(y_train[0])

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                        filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                        activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1, verbose=2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
