import numpy as np

x_ini = np.load("x_ini.npy")
x_true = np.load("x_true.npy")
y_ini = np.load("y_ini.npy")
y_pre1 = np.load("y_pre1.npy")
y_pre2 = np.load("y_pre2.npy")
y_tra1 = np.load("y_tra1.npy")
y_tra2 = np.load("y_tra2.npy")
y_tra21 = np.load("y_tra21.npy")
y_tra22 = np.load("y_tra22.npy")
y_true = np.load("y_true.npy")
y_val1 = np.load("y_val1.npy")
y_val2 = np.load("y_val2.npy")


import matplotlib.pyplot as plt
plt.plot(x_ini,y_ini,label="initial")
plt.plot(x_ini,y_pre1,label="pretrain")
plt.plot(x_ini,y_tra1,label="first train")
plt.plot(x_ini,y_tra21,label="second train")
plt.plot(x_ini,y_val1,label="val")
plt.scatter(x_true, y_true, edgecolors='r')
plt.title("function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

def mse(target, predict):
    k=0
    for i in range(len(target)):
        k+=(target[i] - predict[i])**2
    return k


mse1=mse(y_true,y_pre2)
mse2=mse(y_true,y_tra2)
mse3=mse(y_true,y_tra22)
mse4=mse(y_true,y_val2)
x=[1,2,3,4]
y=[mse1,mse2,mse3,mse4]
plt.plot(x,y,label="MSE")
plt.title("MSE")
plt.xlabel("x")
plt.ylabel("mse")
plt.legend()
plt.show()



plt.scatter(x_true, y_pre2, label="pre")
plt.scatter(x_true, y_tra2, label="tra1")
plt.scatter(x_true, y_tra22, label="tra2")
plt.scatter(x_true, y_val2,label="val")
plt.scatter(x_true, y_true, label="true")
plt.legend()
plt.show()


def dis(target, predict):
    k=[]
    for i in range(len(target)):
        k.append(abs(target[i] - predict[i]))
    return k
dis1=dis(y_true,y_pre2)
dis2=dis(y_true,y_tra2)
dis3=dis(y_true,y_tra22)
dis4=dis(y_true,y_val2)

plt.scatter(x_true, dis1, label="pre")
plt.scatter(x_true, dis2, label="tra1")
plt.scatter(x_true, dis3, label="tra2")
plt.scatter(x_true, dis4,label="val")

plt.title("distance")
plt.xlabel("x")
plt.ylabel("dis")
plt.legend()
plt.show()

