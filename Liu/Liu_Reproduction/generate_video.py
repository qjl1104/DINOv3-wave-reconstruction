import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

# ================= 配置 =================
CONFIG = {
    'data_dir': r'D:\Research\wave_reconstruction_project\output_depth_maps',
    'model_path': r'D:\Research\wave_reconstruction_project\Liu\Liu_Reproduction\checkpoints\wave_model_epoch_50.pth',
    'output_video': r'D:\Research\wave_reconstruction_project\Liu\Liu_Reproduction\wave_reconstruction_result.mp4',
    'img_size': (256, 256),
    'seq_len': 5,
    'fps': 20 # 视频帧率
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 模型定义 (必须与训练时一致) =================
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
    def __init__(self, nf=48):
        super(WaveReconstructionModel, self).__init__()
        self.nf = nf
        self.lstm1 = ConvLSTMCell(input_dim=1, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn1 = nn.BatchNorm2d(nf)
        self.lstm2 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn2 = nn.BatchNorm2d(nf)
        self.lstm3 = ConvLSTMCell(input_dim=nf, hidden_dim=nf, kernel_size=(3,3), bias=True)
        self.bn3 = nn.BatchNorm2d(nf)
        self.final_conv = nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, seq_len, _, h, w = x.size()
        h1, c1 = self.lstm1.init_hidden(b, (h, w))
        h2, c2 = self.lstm2.init_hidden(b, (h, w))
        h3, c3 = self.lstm3.init_hidden(b, (h, w))
        last_output = None
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            h1, c1 = self.lstm1(input_t, (h1, c1))
            h1_bn = self.bn1(h1)
            h2, c2 = self.lstm2(h1_bn, (h2, c2))
            h2_bn = self.bn2(h2)
            h3, c3 = self.lstm3(h2_bn, (h3, c3))
            h3_bn = self.bn3(h3)
            last_output = h3_bn
        out = self.final_conv(last_output)
        out = self.sigmoid(out)
        return out

# ================= 主程序 =================
def run():
    # 1. 加载模型
    print(f"Loading model from {CONFIG['model_path']}...")
    model = WaveReconstructionModel().to(device)
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()

    # 2. 准备数据列表
    files = sorted([f for f in os.listdir(CONFIG['data_dir']) if f.endswith('.png')])
    if len(files) == 0:
        print("Error: No images found!")
        return

    # 3. 初始化视频写入器
    # 输出视频尺寸：宽 = 256*2 (左右拼接), 高 = 256
    video_size = (CONFIG['img_size'][0] * 2, CONFIG['img_size'][1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 或者 'XVID'
    out = cv2.VideoWriter(CONFIG['output_video'], fourcc, CONFIG['fps'], video_size)

    print(f"Generating video: {CONFIG['output_video']}")
    print("Left: Sparse Input (Raw) | Right: Dense Reconstruction (AI)")

    # 4. 逐帧处理
    seq_len = CONFIG['seq_len']
    
    # 我们使用滑动窗口，每张图都预测一次
    for i in tqdm(range(len(files) - seq_len)):
        # 准备输入序列 (5帧)
        seq_imgs = []
        for j in range(seq_len):
            path = os.path.join(CONFIG['data_dir'], files[i + j])
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            seq_imgs.append(img)
        
        # 构造 Tensor [1, 5, 1, 256, 256]
        input_tensor = np.array(seq_imgs)
        input_tensor = np.expand_dims(input_tensor, axis=1) # Add channel
        input_tensor = np.expand_dims(input_tensor, axis=0) # Add batch
        input_tensor = torch.from_numpy(input_tensor).to(device)

        # 推理
        with torch.no_grad():
            pred = model(input_tensor) # [1, 1, 256, 256]
        
        # --- 可视化处理 ---
        # 1. 取出原始输入 (序列的最后一帧)
        raw_img = (seq_imgs[-1] * 255).astype(np.uint8)
        raw_color = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        # 在原始图上加文字
        cv2.putText(raw_color, "Sparse Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 2. 取出预测结果
        pred_img = pred[0, 0].cpu().numpy()
        pred_u8 = (pred_img * 255).astype(np.uint8)
        # 使用伪彩色 (JET Colormap) 渲染波高：蓝色=波谷，红色=波峰
        pred_color = cv2.applyColorMap(pred_u8, cv2.COLORMAP_JET)
        cv2.putText(pred_color, "AI Reconstruction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. 拼接 (左右并排)
        combined = np.hstack((raw_color, pred_color))
        
        # 4. 写入视频
        out.write(combined)

    out.release()
    print("Video saved successfully!")

if __name__ == '__main__':
    run()