import torch
import torch.nn.functional as F
import torch.nn as nn

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        # 创建高斯核
        def gauss(size, sigma=1.5):
            coords = torch.arange(size).float() - size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            return g / g.sum()

        _1D_window = gauss(window_size).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.T
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if size_average else ssim_map

    def forward(self, pred, target):
        # pred, target: [B, 1, H, W], float, [0,1]
        (_, channel, _, _) = pred.size()

        if channel != self.channel:
            self.window = self.create_window(self.window_size, channel)
            self.channel = channel

        window = self.window.to(pred.device)
        return 1 - self.ssim(pred, target, window, self.window_size, channel, self.size_average)

class MultiClassSSIMLoss(nn.Module):
    def __init__(self, window_size=11, focus_classes=None, weight=None):
        super().__init__()
        self.base_ssim = SSIMLoss(window_size)
        self.focus_classes = focus_classes
        self.weight = weight

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        total_loss = 0.0
        num_used = 0

        classes_to_use = self.focus_classes if self.focus_classes else list(range(C))
        weights = self.weight if self.weight else [1.0] * len(classes_to_use)

        for i, c in enumerate(classes_to_use):
            if i == 0:
                continue
            pred_c = pred[:, c:c+1, :, :]     # [B, 1, H, W]
            target_c = target_onehot[:, c:c+1, :, :]
            loss_c = self.base_ssim(pred_c, target_c)
            total_loss += weights[i] * loss_c
            num_used += weights[i]

        return total_loss / num_used