import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio

def add_layer(inputs,in_size,out_size,activation_function="elu"):
    Weights = tf.Variable(tf.random_normal(shape=[in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_b = tf.matmul(tf.cast(inputs,tf.float32),Weights)+biases
    if activation_function==None:
        outputs=Wx_b
    else:
        outputs = activation_function(Wx_b)
    return outputs

def maxminnorm(array):
    maxcols=array.max()
    mincols=array.min()
    data_shape = len(array)

    t=np.empty((data_shape))
    for i in range(data_shape):
        t[i]=(array[i]-mincols)/(maxcols-mincols)
    return t
path=r'/home/liu/下载/points/features20++1.mat'
data = sio.loadmat(path)['AC']
x_data = data[1,:]
x_data=maxminnorm(np.array(x_data))
x_data=x_data[:,np.newaxis]

y_data = data[2,:]
y_data=maxminnorm(np.array(y_data))
y_data=y_data[:,np.newaxis]

xs = tf.placeholder(shape=[None,1],dtype=tf.float32)
ys = tf.placeholder(shape=[None,1],dtype=tf.float32)
#添加隐藏层 和输出层
#l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
l1 = add_layer(xs,1,10,activation_function=tf.nn.sigmoid)
#print('l1:',l1)
prediction = add_layer(l1,10,1,activation_function=None)
#定义损失函数  按列求和 利用梯度下降算法使得loss最小
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#train_step = train_step.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig =plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
fig =plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#进行训练
for i in range(10001):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print('step:',i,sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        #ax.lines.remove(lines[0])
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #先抹除再划线
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        #ax.lines.remove(lines[0])
        plt.pause(0.2)
 
 
