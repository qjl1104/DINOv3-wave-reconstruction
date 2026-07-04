import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
from tensorflow import keras
import os
from keras.optimizers import SGD
class model_x2(tf.keras.Model):
    def __init__(self, opt=False):
        super(model_x2,self).__init__()
        self.no_grad = opt
        self.layer1_1 = tf.keras.layers.Dense(16,activation=tf.nn.relu)
        self.layer1_2 = tf.keras.layers.Dense(16,activation=tf.nn.relu)
        self.layer1_3 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.layer2   = tf.keras.layers.Dense(8,activation="elu")
        self.layer3   = tf.keras.layers.Dense(1,activation="elu")

  
    def call(self,in1):
        
        in3  = tf.keras.layers.Multiply()((in1,in1,in1))
        in2  = tf.keras.layers.Multiply()((in1,in1)) #in1 * in1
        x    = self.layer1_1(in1)
        x1_2 = self.layer1_2(in2)
        x1_3 = self.layer1_3(in3)
        x2   = tf.keras.layers.concatenate((x,x1_2,x1_3))
        if self.no_grad:
            xx = self.layer2(tf.concat((x2, in1.detach()), axis = -1))
        else:
            xx = self.layer2(tf.concat((x2, in1), axis = -1))
        #out1  = self.layer2(x2)
        out=self.layer3(xx)
        return out
def myself_loss(y_true, y_pred):
	losses = abs(y_true - y_pred)
	return losses
model = model_x2()  
model.build(input_shape=(None,1))
model.summary()
 
opt       = SGD(lr=0.001)
los       = tf.keras.losses.MeanSquaredError()
acc       = tf.keras.metrics.MeanSquaredError()
model.compile(optimizer=opt,loss=los,metrics=acc)
 


path=r'/home/liu/下载/wave.mat'
data = sio.loadmat(path)['wave']
x_data = data[0,:]
ds_x=x_data[:,np.newaxis]

y_data = data[1,:]
ds_y=y_data[:,np.newaxis]




checkpoint_path = "/home/liu/下载/fitting/fitting/model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(ds_x,ds_y,epochs=500,batch_size=16, callbacks=[cp_callback])
x =x_data
x=x[:,np.newaxis]
 
y_predict = model.predict(x)
 
ds_x = x_data
ds_y = y_data
 
plt.scatter(ds_x,ds_y)
plt.plot(x,y_predict,'r')
plt.show()