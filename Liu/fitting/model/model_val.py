from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import scipy.io as sio
import matplotlib.pyplot as plt
def maxminnorm(array):
    maxcols=array.max()
    mincols=array.min()
    data_shape = len(array)

    t=np.empty((data_shape))
    for i in range(data_shape):
        t[i]=(array[i]-mincols)/(maxcols-mincols)
    return t,maxcols,mincols
path=r'/home/liu/下载/points/features20++1.mat'
data = sio.loadmat(path)['AC']
x_data = data[1,:]
x_data,maxx,minx=maxminnorm(np.array(x_data))
ds_x=x_data[:,np.newaxis]

y_data = data[2,:]
y_data=y_data-np.mean(y_data)
y_data,maxy,miny=maxminnorm(np.array(y_data))
ds_y=y_data[:,np.newaxis]

path1=r'/home/liu/下载/wave1.mat'
path2=r'/home/liu/下载/wave2.mat'
path3=r'/home/liu/下载/wave3.mat'
data1 = sio.loadmat(path1)['wave']
data2 = sio.loadmat(path2)['wave']
data3 = sio.loadmat(path3)['wave']




dataset=TensorDataset(torch.tensor(ds_x,dtype=torch.float),torch.tensor(ds_y,dtype=torch.float))

dataloader=DataLoader(dataset,batch_size=20,shuffle=True)



# 神经网络主要结构，这里就是一个简单的线性结构
class FGLayer(nn.Module):
    def __init__(self, in_size, out_size, opt=False):
        super().__init__()
        self.no_grad = opt
        self.active = nn.ReLU()
        self.layer1 = torch.nn.Sequential(
                        torch.nn.Linear(in_size, out_size),
                        torch.nn.ReLU(),
                        )
        self.layer2 = torch.nn.Sequential(
                        torch.nn.Linear(in_size + out_size, out_size),
                        torch.nn.Sigmoid(),
                        )
        self.layer3 = torch.nn.Sequential(
                torch.nn.Linear(out_size, out_size),
                torch.nn.ELU(),
                )
    def forward(self, input):
        mid_inf = self.layer1(input)
        if self.no_grad:
            gate_inf = self.layer2(torch.cat((mid_inf, input.detach()), dim = -1))
        else:
            gate_inf = self.layer2(torch.cat((mid_inf, input), dim = -1))
        out=self.layer3(gate_inf*mid_inf)
        return out
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.net=nn.Sequential(
        FGLayer(1,100),
        FGLayer(100, 100),
        nn.Linear(100, 1)
    )


  def forward(self, input:torch.FloatTensor):
    return self.net(input)

net=Net()
state_dict = torch.load('only_weights2.pth')
net.load_state_dict(state_dict)
'''
path2="/home/liu/下载/fitting/fitting/model/only_weights.pth"
net.load_state_dict(torch.load(path2))#加载参数
net.eval()
'''
# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),lr=0.01)
Loss=nn.MSELoss()

train_losses = []
train_acces = []
# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
  train_loss = 0
  train_acc = 0


  loss=None
  for batch_x,batch_y in dataloader: 

    y_predict=net(batch_x)

    loss=Loss(y_predict,batch_y)


    optim.zero_grad()

    loss.backward()
    
    optim.step()
    out_t = y_predict.argmax(dim=1)
    num_correct = (out_t == batch_x).sum().item()
    acc = num_correct / batch_x.shape[0]
    train_acc += acc
    train_loss += loss.item()
    #train_loss = torch.tensor(train_loss, device = 'cpu')
    #train_acc = torch.tensor(train_acc, device = 'cpu')
  train_losses.append(train_loss)
  train_acces.append(train_acc)
  # 每100次 的时候打印一次日志
  if (epoch+1)%100==0:
    print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
#train_acces = torch.tensor([item.cpu().detach().numpy() for item in train_acces]).cuda()
#train_losses = torch.tensor([item.cpu().detach().numpy() for item in train_losses]).cuda()

# 使用训练好的模型进行预测
factx=[2570.5,480.9,586.3,372.3,649.3,1128.8,1704.9,911]
fx=(factx-minx)/(maxx-minx)
fx=fx[:,np.newaxis]
fy=[42.4,9.5,1.1,20.8,-3,-40,-20.6,-25.6]
predict=net(torch.tensor(ds_x,dtype=torch.float))
predict2=net(torch.tensor(fx,dtype=torch.float))
p1=predict.detach().numpy()
p2=predict2.detach().numpy()
p_p1=p1*(maxy-miny)+miny
p_p2=p2*(maxy-miny)+miny


np.save("y_val1.npy", p_p1)
np.save("y_val2.npy", p_p2)


torch.save(net.state_dict(), 'only_weights3.pth')
torch.save(net, "my_model3.pth")


ds_x=ds_x*(maxx-minx)+minx
ds_y=ds_y*(maxy-miny)+miny
# 绘图展示预测的和真实数据之间的差异
import matplotlib.pyplot as plt
plt.plot(ds_x,ds_y,label="fact")
plt.plot(ds_x,(predict.detach().numpy())*(maxy-miny)+miny,label="predict")
plt.scatter(factx, fy, edgecolors='r')
plt.title("function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()


#plt.plot(np.arange(len(train_losses)), train_losses,label="train loss")

plt.plot(np.arange(len(train_acces)), train_acces, label="train acc")


plt.legend() #显示图例
plt.xlabel('epoches')
#plt.ylabel("epoch")
plt.title('Model accuracy&loss')
plt.show()