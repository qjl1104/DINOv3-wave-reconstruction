#导入所需的库
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio

path=r'/home/liu/下载/points/features20++1.mat'
data = sio.loadmat(path)['AC']
x = data[1,:]
y = data[2,:]
x,y=x[0:200],y[0:200]
X = np.expand_dims(x,axis = 1) #转变x,y的输出
Y = y.reshape(len(y),-1) #X,Y输出均为(400,1)
#把独立的数据利用Pytorch提供的TensorDataset转化为tensor类型的数据集
dataset = TensorDataset(torch.tensor(X,dtype=torch.float),torch.tensor(Y,dtype=torch.float))
#分批加载数据集中的数据到内存/显存
dataloader = DataLoader(dataset,batch_size = 200,shuffle = True)

#设置网络模型，设定三个隐藏层、一个输出层
 
class Net(nn.Module): #定义网络模型
    def __init__(self): #初始化
        super(Net,self).__init__() #父类初始化
        self.net = nn.Sequential(
                                nn.Linear(in_features=1,out_features=11),nn.ReLU(),
                                nn.Linear(11,120),nn.ReLU(),
                                nn.Linear(120,10),nn.ReLU(),
                                nn.Linear(10,1)    
        )
        
    def forward(self,input:torch.FloatTensor):
        return self.net(input)
net=Net()
net
Loss = nn.MSELoss()
optim = optim.Adam(net.parameters(),lr = 0.001)

for epoch in range(1000):
    loss = None
    for batch_x,batch_y in dataloader:
        y_predict = net(batch_x)
        loss = Loss(y_predict,batch_y)
        optim.zero_grad()  #在进行新一轮训练之前，把上一轮用过的梯度清空
        loss.backward()    #损失反向传播，得到新的网络参数
        optim.step()       #把上一轮得到的新的网络参数更新到网络模型中
        
    if(epoch+1)%10==0:
        print('训练次数:{0},模型损失:{1}'.format(epoch+1,loss.item())) #.item把仅含一个元素的张量中的值拿出来
predict=net(torch.tensor(X,dtype=torch.float))
plt.figure(figsize=(12,8))
plt.plot(x,y,label='real value',marker = 'o')
plt.plot(x,predict.detach().numpy(),label='predict value',marker = '*',c = 'r')
plt.legend()
plt.show()