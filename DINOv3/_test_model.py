import sys
sys.path.insert(0, '.')
from config import Config
from models import CorrMatchingStereoModel
import torch

cfg = Config()
model = CorrMatchingStereoModel(cfg)
print('Model created successfully')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

B = 2
feat_l = torch.randn(B, 768, 100, 160)
feat_r = torch.randn(B, 768, 100, 160)
kpl = torch.rand(B, 50, 2) * torch.tensor([2560, 1600]).float()
sl = torch.ones(B, 50)
kpr = torch.rand(B, 60, 2) * torch.tensor([2560, 1600]).float()
sr = torch.ones(B, 60)
lg = torch.rand(B, 1, 1600, 2560)
rg = torch.rand(B, 1, 1600, 2560)
mask = torch.ones(B, 1, 1600, 2560)

cached_data = {
    'feat_left': feat_l,
    'feat_right': feat_r,
    'keypoints_left': kpl,
    'scores_left': sl,
    'keypoints_right': kpr,
    'scores_right': sr,
}

with torch.no_grad():
    out = model(lg, rg, lg.repeat(1,3,1,1), rg.repeat(1,3,1,1), mask, cached_data=cached_data)

print(f"Output keys: {list(out.keys())}")
print(f"Disparity shape: {out['disparity'].shape}")
print(f"Disparity range: [{out['disparity'].min():.1f}, {out['disparity'].max():.1f}]")
print(f"kp_right_pred shape: {out['keypoints_right_pred'].shape}")
print('Forward pass OK!')
