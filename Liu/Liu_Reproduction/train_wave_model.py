import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json  # 用于保存训练历史

# ===============================================================================
# 1. 配置参数
# ===============================================================================
CONFIG = {
    # 输入数据路径：改回原始路径 (源数据位置)
    'data_dir': r'D:\Research\wave_reconstruction_project\output_depth_maps',
    
    'img_size': (256, 256),  # 论文输入尺寸
    'seq_len': 5,            # 输入序列长度 (利用过去5帧预测)
    'batch_size': 8,
    'learning_rate': 1e-3,
    'epochs': 50,
    
    # 输出保存路径：保存在 Liu_Reproduction 子目录下
    'save_dir': r'D:\Research\wave_reconstruction_project\Liu\Liu_Reproduction\checkpoints'
}

# 自动创建保存目录
if not os.path.exists(CONFIG['save_dir']):
    os.makedirs(CONFIG['save_dir'])
    print(f"Created checkpoint directory: {CONFIG['save_dir']}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===============================================================================
# 2. ConvLSTM 模块定义 (复现论文结构)
# ===============================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class WaveReconstructionModel(nn.Module):
    """
    对应论文表 4.2 的结构:
    ConvLSTM (48) -> BN -> ConvLSTM (48) -> BN -> ConvLSTM (48) -> BN -> Conv2D (1)
    """
    def __init__(self, nf=48):
        super(WaveReconstructionModel, self).__init__()
        
        self.nf = nf
        # 三层 ConvLSTM
        self.lstm1 = ConvLSTMCell(input_dim=1, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn1 = nn.BatchNorm2d(nf)
        self.lstm2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn2 = nn.BatchNorm2d(nf)
        self.lstm3 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn3 = nn.BatchNorm2d(nf)
        
        # 输出层还原为灰度图
        self.final_conv = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [Batch, Seq, Channel, H, W]
        b, seq_len, _, h, w = x.size()
        
        # 初始化状态
        h1, c1 = self.lstm1.init_hidden(b, (h, w))
        h2, c2 = self.lstm2.init_hidden(b, (h, w))
        h3, c3 = self.lstm3.init_hidden(b, (h, w))
        
        last_output = None
        
        # 时序处理
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            
            h1, c1 = self.lstm1(input_t, (h1, c1))
            h1_bn = self.bn1(h1)
            
            h2, c2 = self.lstm2(h1_bn, (h2, c2))
            h2_bn = self.bn2(h2)
            
            h3, c3 = self.lstm3(h2_bn, (h3, c3))
            h3_bn = self.bn3(h3)
            
            last_output = h3_bn
            
        # 最后一帧输出
        out = self.final_conv(last_output)
        out = self.sigmoid(out) # 归一化到 [0, 1]
        return out

# ===============================================================================
# 3. 数据集加载 (Target 稠密化)
# ===============================================================================

class WaveDataset(Dataset):
    def __init__(self, data_dir, seq_len=5):
        self.data_dir = data_dir
        self.seq_len = seq_len
        # 增加鲁棒性：检查目录是否存在
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}，请检查路径配置。")
            
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        print(f"Found {len(self.files)} images in {data_dir}")

    def __len__(self):
        return len(self.files) - self.seq_len

    def densify(self, img):
        """
        【关键步骤】将稀疏的骨架图转化为稠密的波浪面
        用于生成 Ground Truth (y)，指导模型学习“填补空洞”
        """
        # 转换到 0-255 uint8
        img_u8 = (img * 255).astype(np.uint8)
        
        # 1. 闭运算：连接断裂的特征点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_u8 = cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 2. 膨胀：让波浪“长肉”
        img_u8 = cv2.dilate(img_u8, kernel, iterations=2)
        
        # 3. 高斯模糊：平滑化，模拟连续水面
        img_u8 = cv2.GaussianBlur(img_u8, (21, 21), 0)
        
        return img_u8.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        # 读取序列: [t, t+1, ..., t+seq_len-1]
        seq_imgs = []
        for i in range(self.seq_len):
            img_path = os.path.join(self.data_dir, self.files[idx + i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 简单的预处理
            if img is None:
                # 容错处理
                img = np.zeros(CONFIG['img_size'], dtype=np.uint8)
                
            img = img.astype(np.float32) / 255.0
            seq_imgs.append(img)
            
        data = np.array(seq_imgs) # [Seq, H, W]
        data = np.expand_dims(data, axis=1) # [Seq, 1, H, W]
        
        # Input (x): 保持稀疏，这是模型在实际中看到的
        x = torch.from_numpy(data)
        
        # Target (y): 进行稠密化处理，这是我们希望模型输出的
        target_frame = data[-1, 0] # [H, W]
        dense_target = self.densify(target_frame)
        dense_target = np.expand_dims(dense_target, axis=0) # [1, H, W]
        
        y = torch.from_numpy(dense_target) 
        
        return x, y

# ===============================================================================
# 4. 训练主循环
# ===============================================================================

def train():
    dataset = WaveDataset(CONFIG['data_dir'], seq_len=CONFIG['seq_len'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    
    model = WaveReconstructionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss() 

    print("Start training (Sparse Input -> Dense Output)...")
    loss_history = {'train_loss': []}  # 用于记录训练历史

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device) # [B, Seq, 1, H, W] - 稀疏
            y = y.to(device) # [B, 1, H, W] - 稠密
            
            optimizer.zero_grad()
            output = model(x)
            
            # 使用 MSE Loss
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history['train_loss'].append(avg_loss)
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {avg_loss:.6f}")
        
        # 保存模型和预览图 (每5轮)
        if (epoch + 1) % 5 == 0: 
            # 1. 保存模型权重
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], f'wave_model_epoch_{epoch+1}.pth'))
            
            # 2. 保存训练历史数据 (JSON)
            with open(os.path.join(CONFIG['save_dir'], 'history.json'), 'w') as f:
                json.dump(loss_history, f)
            
            # 3. 增强可视化: 使用 'jet' 颜色映射
            with torch.no_grad():
                input_sparse = x[0, -1, 0].cpu().numpy()
                target_dense = y[0, 0].cpu().numpy()
                pred = output[0, 0].cpu().numpy()
                
                plt.figure(figsize=(18, 5))
                
                # 第一张：稀疏输入
                ax1 = plt.subplot(1, 3, 1)
                plt.title("Input (Sparse)")
                im1 = plt.imshow(input_sparse, cmap='gray', vmin=0, vmax=1)
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                
                # 第二张：稠密目标 (Dense GT)
                ax2 = plt.subplot(1, 3, 2)
                plt.title("Target (Densified GT)")
                im2 = plt.imshow(target_dense, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                
                # 第三张：模型预测 (Reconstructed)
                ax3 = plt.subplot(1, 3, 3)
                plt.title("Prediction (Learned)")
                im3 = plt.imshow(pred, cmap='jet', vmin=0, vmax=1)
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
                
                plt.savefig(os.path.join(CONFIG['save_dir'], f'val_epoch_{epoch+1}.png'))
                plt.close()

    # 训练结束后，绘制 Loss 曲线
    print("Training finished!")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history['train_loss']) + 1), loss_history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'))
    plt.close()

if __name__ == '__main__':
    train()