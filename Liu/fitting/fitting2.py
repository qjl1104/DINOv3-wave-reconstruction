import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.io as sio

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # shape
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
rnn = RNN().cuda()

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_func = nn.MSELoss()

plt.ion()
plt.show()
plt.figure(figsize=(12,6))

h_state = None
path=r'/home/liu/下载/wave.mat'
data = sio.loadmat(path)['wave']
x_data = data[0,:]
x_np=x_data[:,np.newaxis]

y_data = data[1,:]
y_np=y_data[:,np.newaxis]
x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis])).cuda() # shape (batch, time_step, input_size)
y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis])).cuda()

prediction, h_state = rnn(x, h_state)
h_state = Variable(h_state.data).cuda()
loss = loss_func(prediction, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('loss=%.2f' % loss)
plt.plot(x_np, y_np, 'r-', lw=1)
plt.plot(x_np, prediction.cpu(), 'b-', lw=1)
plt.pause(0.2)

plt.ioff()
plt.show()
