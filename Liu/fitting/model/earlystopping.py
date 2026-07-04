# earlystopping.py
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience  # 等待多少个epoch之后停止
        self.verbose = verbose  # 是否显示日志
        self.counter = 0  # 计步器
        self.best_score = None  # 记录最好性能
        self.early_stop = False  # 早停触发
        self.val_psnr_min = 0  # 记录最小的验证PSNR
        self.delta = delta  # 可以给最好性能加上的小偏置
        self.checkpoint_perf = []  # 记录检查点的性能

    def __call__(self, g, d, train_psnr, val_psnr):

        score = val_psnr
        self.early_stop = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)
        elif score < self.best_score + self.delta:  # PSNR越大越好，因此这里是小于，若使用loss做指标，这里应改成大于
            self.counter += 1  # 若当前性能不超过前一个epoch的性能则计步器+1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # 计步器累计到达极限，出发早停
                self.early_stop = True
                self.counter = 0
                self.best_score = None
                self.val_psnr_min = 0
        else:  # 当前性能优于或等于前一个epoch的性能，则更新最佳性能记录
            self.best_score = score
            self.save_checkpoint(g, d, val_psnr)  # 保存检查点
            self.counter = 0  # 计步器重置
            self.checkpoint_perf = [train_psnr, val_psnr]  # 记录检查点性能数据
        return self.checkpoint_perf

    def save_checkpoint(self, g, d, val_psnr):  # 保存检查点
        self.val_psnr_min = val_psnr
        if self.verbose:
            print(f'Validation PSNR increased ({self.val_psnr_min:.6f} --> {val_psnr:.6f}).  Saving model ...')
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')
        else:
            torch.save(g.state_dict(), 'Generator.pth')
            torch.save(d.state_dict(), 'Discriminator.pth')
