import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio

path=r'/home/liu/下载/points/features20++1.mat'
data = sio.loadmat(path)['AC']
x = data[1,:]
y = data[2,:]


# 定义网络架构
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=(1,),activation="elu"),
    tf.keras.layers.Dense(1)
])

#设置训练参数
model.compile(
    optimizer="adam",
    loss="mse"
)

#训练并查看训练进度
history = model.fit(x,y,epochs=5000)
y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x,y_predict,'r')
plt.show()


