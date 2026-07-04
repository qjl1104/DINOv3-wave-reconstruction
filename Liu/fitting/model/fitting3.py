import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio


def maxminnorm(array):
    maxcols = array.max()
    mincols = array.min()
    data_shape = len(array)

    t = np.empty((data_shape))
    for i in range(data_shape):
        t[i] = (array[i] - mincols) / (maxcols - mincols)
    return t


path = r'/home/liu/下载/points/features20++1.mat'
data = sio.loadmat(path)['AC']
x_data = data[1, :]
x_data = maxminnorm(np.array(x_data))
ds_x = x_data[:, np.newaxis]

y_data = data[2, :]
y_data = y_data - np.mean(y_data)
y_data = maxminnorm(np.array(y_data))
ds_y = y_data[:, np.newaxis]


class model_x2(tf.keras.Model):
    def __init__(self):
        super(model_x2, self).__init__()
        self.layer1 = tf.keras.layers.Dense(1, activation="elu")
        self.layer1_2 = tf.keras.layers.Dense(2, activation="elu")
        self.layer1_3 = tf.keras.layers.Dense(2, activation="elu")
        self.layer2 = tf.keras.layers.Dense(1, activation="elu")

    def call(self, in1):
        in2 = tf.keras.layers.Multiply()((in1, in1, in1))
        in3 = tf.keras.layers.Multiply()((in1, in1))  # in1 * in1
        x = self.layer1(in2)
        x1_2 = self.layer1_2(in1)
        x1_3 = self.layer1_3(in3)
        x2 = tf.keras.layers.concatenate((x, x1_2, x1_3))
        out = self.layer2(x2)
        return out


model = model_x2()
model.build(input_shape=(None, 1))
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
los = tf.keras.losses.MeanSquaredError()
acc = tf.keras.metrics.MeanSquaredError()
model.compile(optimizer=opt, loss=los, metrics=acc)

model.fit(ds_x, ds_y, epochs=1000)

x = x_data
x = x[:, np.newaxis]

y_predict = model.predict(x)

ds_x = x_data
ds_y = y_data

plt.scatter(ds_x, ds_y)
plt.plot(x, y_predict, 'r')
plt.show()
