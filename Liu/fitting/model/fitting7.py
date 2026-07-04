from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
from torch.nn.parameter import Parameter
import scipy.io as sio
from torchsummary import summary
import matplotlib.pyplot as plt

path=r'/home/liu/下载/wave.mat'
data = sio.loadmat(path)['wave']
x_data = data[0,:]
x_data=x_data-np.mean(x_data)
ds_x=x_data[:,np.newaxis]

y_data = data[1,:]
ds_y=y_data[:,np.newaxis]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset=TensorDataset(torch.tensor(ds_x,dtype=torch.float).to(device),torch.tensor(ds_y,dtype=torch.float).to(device))
dataloader=DataLoader(dataset,batch_size=300,shuffle=True)

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
    return self.net(input.to(device))

net=Net().to(device)
summary(net, (100, 1))
# 定义优化器和损失函数
optim=torch.optim.Adam(Net.parameters(net),lr=0.001)
Loss=nn.MSELoss()

# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
  loss=None
  for batch_x,batch_y in dataloader:
    batch_x=batch_x.to(device)
    batch_y=batch_y.to(device)
    y_predict=net(batch_x)
    loss=Loss(y_predict,batch_y)
    optim.zero_grad()
    loss.backward()


    optim.step()
  # 每100次 的时候打印一次日志
  if (epoch+1)%100==0:
    print("step: {0} , loss: {1}".format(epoch+1,loss.item()))

# 使用训练好的模型进行预测
predict=net(torch.tensor(ds_x,dtype=torch.float))
predict=predict.cpu()
torch.save(net.state_dict(), 'only_weights.pth')
torch.save(net, "my_model.pth")
# 绘图展示预测的和真实数据之间的差异
import matplotlib.pyplot as plt
plt.plot(ds_x,ds_y,label="fact")
plt.plot(ds_x,predict.detach().numpy(),label="predict")
plt.title("function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.show()

