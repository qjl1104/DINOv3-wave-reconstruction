"""
DINOv3 Wave Reconstruction - Training Script
==============================================
V24.10 architecture with improvements:
  - Shared model/config/loss/dataset modules
  - CosineAnnealingLR scheduler
  - Validation set evaluation
  - Best model saving (by val loss)
  - Full checkpoint (model + optimizer + scheduler + scaler)
  - Random seed for reproducibility
"""

import os
import sys
import json
import random
import traceback
from datetime import datetime
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Shared modules ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config, check_path
from models import SparseMatchingStereoModel
from losses import PINNPhysicsLoss
from dataset import RectifiedWaveStereoDataset, stereo_collate_fn


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Random seed set to {seed}")


class Trainer:
    """Training manager with validation, LR scheduling, and checkpointing."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = os.path.join(cfg.RUNS_BASE_DIR, self.timestamp)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.vis_dir = os.path.join(self.run_dir, "vis")
        for d in [self.ckpt_dir, self.log_dir, self.vis_dir]:
            os.makedirs(d, exist_ok=True)

        # --- Model ---
        print(f"--- 初始化模型 ---")
        self.model = SparseMatchingStereoModel(cfg).to(self.device)

        # Optional pretrained loading
        if cfg.PRETRAINED_CHECKPOINT:
            if os.path.exists(cfg.PRETRAINED_CHECKPOINT):
                self._load_pretrained(cfg.PRETRAINED_CHECKPOINT)
            else:
                print(f"[Warning] 预训练 checkpoint 未找到: {cfg.PRETRAINED_CHECKPOINT}")
                print("[Warning] 将从头开始训练...")
        else:
            print("[Info] 未指定预训练 checkpoint，从头开始训练。")

        # --- Optimizer + Scheduler ---
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.NUM_EPOCHS
        )
        self.loss_fn = PINNPhysicsLoss(cfg)
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)

        # --- Datasets ---
        print(f"--- 初始化数据集 ---")
        train_ds = RectifiedWaveStereoDataset(cfg, is_validation=False)
        val_ds = RectifiedWaveStereoDataset(cfg, is_validation=True)
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
            collate_fn=stereo_collate_fn, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            collate_fn=stereo_collate_fn, num_workers=0
        )

        # --- Tracking ---
        self.history = {
            'total_loss': [], 'photo_loss': [], 'epi_loss': [],
            'smooth_loss': [], 'slope_loss': [], 'mean_loss': [],
            'val_loss': [], 'lr': [],
        }
        self.best_val_loss = float('inf')
        self.log_file = os.path.join(self.log_dir, "training_log.json")

        # Save config
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(asdict(cfg), f, indent=2)

    def _load_pretrained(self, path):
        """Load pretrained checkpoint (supports both old and new formats)."""
        print(f"[Pretrained] 正在加载: {path}")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"[Pretrained] 已加载新格式 checkpoint (epoch {checkpoint.get('epoch', '?')})")
            else:
                self.model.load_state_dict(checkpoint, strict=False)
                print("[Pretrained] 已加载旧格式 checkpoint (纯 state_dict)")
        except Exception as e:
            print(f"[致命错误] 模型加载失败: {e}")
            sys.exit(1)

    def _pad_to_14(self, *tensors):
        """Pad tensors to be divisible by 14 (DINOv2 patch size)."""
        h, w = tensors[0].shape[2:]
        padh = (14 - h % 14) % 14
        padw = (14 - w % 14) % 14
        if padh > 0 or padw > 0:
            return [F.pad(t, (0, padw, 0, padh)) for t in tensors]
        return list(tensors)

    def _compute_loss(self, batch):
        """Run forward pass and compute all losses. Returns (total_loss, loss_dict)."""
        lg = batch['left_gray'].to(self.device)
        rg = batch['right_gray'].to(self.device)
        lrgb = batch['left_rgb'].to(self.device)
        rrgb = batch['right_rgb'].to(self.device)
        mask = batch['mask'].to(self.device)
        Q = batch['Q'].to(self.device)

        lg, rg, lrgb, rrgb, mask = self._pad_to_14(lg, rg, lrgb, rrgb, mask)

        with torch.amp.autocast('cuda', enabled=True):
            out = self.model(lg, rg, lrgb, rrgb, mask, apply_epipolar_mask=False)
            kpl = out['keypoints_left']
            kpr_pred = out['keypoints_right_pred']
            scores = out['scores_left']

            l_photo, l_epi, l_smooth, l_slope, l_zeromean = self.loss_fn(
                lg, rg, kpl, kpr_pred, scores, Q
            )

            w_photo = self.cfg.PHOTOMETRIC_WEIGHT * l_photo
            w_epi = self.cfg.EPIPOLAR_WEIGHT * l_epi
            w_smooth = self.cfg.PHY_SMOOTH_WEIGHT * l_smooth
            w_slope = self.cfg.PHY_SLOPE_WEIGHT * l_slope
            w_zero = self.cfg.PHY_ZEROMEAN_WEIGHT * l_zeromean

            total = w_photo + w_epi + w_smooth + w_slope + w_zero

        loss_dict = {
            'total': total.item(),
            'photo': l_photo.item(),
            'epi': l_epi.item(),
            'smooth': l_smooth.item(),
            'slope': l_slope.item(),
            'mean': l_zeromean.item(),
        }
        return total, loss_dict

    def sanity_check(self):
        """Run a single forward pass to verify model state before training."""
        print("\n[Self-Check] 正在运行自检...")
        self.model.eval()
        try:
            batch = next(iter(self.train_loader))
            if batch is None:
                print("[Self-Check] 无法加载数据批次")
                return

            with torch.no_grad():
                total, loss_dict = self._compute_loss(batch)

            print(f"[Self-Check] Initial Loss: {total.item():.4f}")
            print(f"    - Photo: {loss_dict['photo']:.4f}")
            print(f"    - Epi:   {loss_dict['epi']:.4f}")
            print(f"    - Phy:   {loss_dict['smooth']:.4f} + {loss_dict['slope']:.4f} + {loss_dict['mean']:.4f}")

            if total.item() > 50.0:
                print("\n[警告] 初始 Loss 异常高 (>50)！检查模型/数据。")
            else:
                print("[Self-Check] 状态良好。")

        except Exception as e:
            print(f"[Self-Check] 自检失败: {e}")
            traceback.print_exc()

    def _train_one_epoch(self, epoch):
        """Run one training epoch. Returns average loss dict."""
        self.model.train()
        ep_stats = {k: 0.0 for k in ['total', 'photo', 'epi', 'smooth', 'slope', 'mean']}
        count = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.NUM_EPOCHS}")
        for step, batch in enumerate(pbar):
            if batch is None:
                continue

            total_loss, loss_dict = self._compute_loss(batch)
            scaled_loss = total_loss / self.cfg.ACCUMULATION_STEPS

            self.scaler.scale(scaled_loss).backward()

            if (step + 1) % self.cfg.ACCUMULATION_STEPS == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k in ep_stats:
                ep_stats[k] += loss_dict[k]
            count += 1
            pbar.set_postfix({
                'Loss': f"{loss_dict['total']:.1f}",
                'Photo': f"{loss_dict['photo']:.3f}",
                'Sm': f"{loss_dict['smooth']:.2f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        if count > 0:
            return {k: v / count for k, v in ep_stats.items()}
        return ep_stats

    @torch.no_grad()
    def _validate(self, epoch):
        """Run validation. Returns average total loss."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch in self.val_loader:
            if batch is None:
                continue
            loss, _ = self._compute_loss(batch)
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else float('inf')
        print(f"[Val] Epoch {epoch + 1}: Val Loss = {avg_loss:.4f}")
        return avg_loss

    def _save_checkpoint(self, epoch, is_best=False):
        """Save full checkpoint."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.cfg),
        }
        if is_best:
            path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(state, path)
            print(f"[Checkpoint] Best model saved: {path} (val_loss={self.best_val_loss:.4f})")
        else:
            path = os.path.join(self.ckpt_dir, f"model_ep{epoch + 1}.pth")
            torch.save(state, path)
            print(f"[Checkpoint] Saved: {path}")

    def update_json_log(self, epoch):
        """Write training log to JSON."""
        data = {'epoch': epoch, 'history': self.history, 'config': asdict(self.cfg)}
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write JSON log: {e}")

    def plot_history(self):
        """Generate loss history plots."""
        epochs = range(1, len(self.history['total_loss']) + 1)
        plt.figure(figsize=(20, 5))

        # Total Loss
        plt.subplot(1, 4, 1)
        plt.plot(epochs, self.history['total_loss'], label='Total', color='black')
        plt.title('Total Loss')
        plt.grid(True, alpha=0.3)

        # Geometric
        plt.subplot(1, 4, 2)
        plt.plot(epochs, self.history['photo_loss'], label='Photo', color='blue')
        plt.plot(epochs, self.history['epi_loss'], label='Epipolar', color='purple')
        plt.title('Geometric')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Physics
        plt.subplot(1, 4, 3)
        plt.plot(epochs, self.history['smooth_loss'], label='Smooth', color='green')
        plt.plot(epochs, self.history['slope_loss'], label='Slope', color='red')
        plt.plot(epochs, self.history['mean_loss'], label='ZeroMean', color='orange')
        plt.title('Physics')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # LR + Val
        plt.subplot(1, 4, 4)
        if self.history['val_loss']:
            val_epochs = [i * 5 for i in range(1, len(self.history['val_loss']) + 1)]
            plt.plot(val_epochs, self.history['val_loss'], 'ro-', label='Val Loss')
        plt.plot(epochs, self.history['lr'], 'b--', label='LR', alpha=0.5)
        plt.title('Validation & LR')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "loss_history.png"))
        plt.close()

    def train(self):
        """Main training loop."""
        self.sanity_check()
        print(f"\n--- 开始训练 ({self.cfg.NUM_EPOCHS} epochs) ---")

        for epoch in range(self.cfg.NUM_EPOCHS):
            # Train
            avg_losses = self._train_one_epoch(epoch)

            # Record history
            self.history['total_loss'].append(avg_losses['total'])
            self.history['photo_loss'].append(avg_losses['photo'])
            self.history['epi_loss'].append(avg_losses['epi'])
            self.history['smooth_loss'].append(avg_losses['smooth'])
            self.history['slope_loss'].append(avg_losses['slope'])
            self.history['mean_loss'].append(avg_losses['mean'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Step LR scheduler
            self.scheduler.step()

            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                val_loss = self._validate(epoch)
                self.history['val_loss'].append(val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # Save log
            self.update_json_log(epoch)

            # Plot
            if (epoch + 1) % self.cfg.VISUALIZE_INTERVAL == 0:
                self.plot_history()

            # Regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        print(f"\n--- 训练完成 ---")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Run directory: {self.run_dir}")


if __name__ == "__main__":
    try:
        cfg = Config()
        set_seed(cfg.SEED)
        trainer = Trainer(cfg)
        trainer.train()
    except Exception as e:
        print("\n" + "=" * 40)
        print("训练发生错误:")
        print(e)
        print("=" * 40)
        print(traceback.format_exc())