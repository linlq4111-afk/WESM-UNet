import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HausdorffDTLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    def forward(self, probs, targets):
        # probs: [B, 1, H, W] (float, 0~1), targets: [B, 1, H, W] (float, 0/1)
        assert probs.shape == targets.shape
        batch_size = probs.shape[0]
        total_loss = 0.0
        for i in range(batch_size):
            p = probs[i, 0].detach().cpu().numpy()
            t = targets[i, 0].detach().cpu().numpy()
            dt_p = self._distance_transform(1 - p)
            dt_t = self._distance_transform(1 - t)
            # 归一化距离图，防止loss爆炸
            dt_p = dt_p / (dt_p.max() + 1e-8)
            dt_t = dt_t / (dt_t.max() + 1e-8)
            dt_p = torch.from_numpy(dt_p).float().to(probs.device)
            dt_t = torch.from_numpy(dt_t).float().to(probs.device)
            loss = torch.mean((torch.abs(probs[i, 0] - targets[i, 0]) ** self.alpha) * (dt_p ** self.alpha + dt_t ** self.alpha))
            total_loss += loss
        return total_loss / batch_size

    @staticmethod
    def _distance_transform(binary_img):
        # binary_img: numpy array, 0~1
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(binary_img) 