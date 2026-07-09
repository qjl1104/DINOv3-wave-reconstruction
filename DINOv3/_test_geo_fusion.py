"""测试 DINO + 几何指纹融合模型能否正常创建和前向传播。"""
import sys
sys.path.insert(0, 'd:/Research/wave_reconstruction_project/DINOv3')

from config import Config
import torch
from models import CorrMatchingStereoModel

cfg = Config()
print('GEO_KNN_K:', cfg.GEO_KNN_K)
print('GEO_FUSION_DIM:', cfg.GEO_FUSION_DIM)

print('Creating model...')
model = CorrMatchingStereoModel(cfg)
print('Model created successfully')

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params: {total:,}')
print(f'Trainable params: {trainable:,}')

print('Testing forward pass...')
model = model.cuda()
lg = torch.randn(1, 1, 1600, 2560, device='cuda')
rg = torch.randn(1, 1, 1600, 2560, device='cuda')
lrgb = torch.randn(1, 3, 1600, 2560, device='cuda')
rrgb = torch.randn(1, 3, 1600, 2560, device='cuda')
mask = torch.ones(1, 1, 1600, 2560, device='cuda')

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(lg, rg, lrgb, rrgb, mask)

print('Output keys:', list(out.keys()))
print('keypoints_left shape:', out['keypoints_left'].shape)
print('disparity shape:', out['disparity'].shape)
print('correlation_probs count:', len(out['correlation_probs']))
if out['correlation_probs']:
    print('First prob shape:', out['correlation_probs'][0].shape)
print('SUCCESS')