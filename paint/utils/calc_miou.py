import numpy as np


class IoUMetrics:
    def __init__(self, num_classes, ignore_background=True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.reset()

    def reset(self):
        # Micro计算需要的累计混淆矩阵
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        更新IoU指标
        pred: [H, W] 预测类别标签
        target: [H, W] 真实类别标签
        """
        # 更新混淆矩阵（用于micro计算）
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist


    def compute_micro_iou(self):
        """Micro mIoU：基于全局混淆矩阵计算"""
        ious = []
        start_class = 1 if self.ignore_background else 0

        for i in range(start_class, self.num_classes):
            tp = self.confusion_matrix[i, i]  # True Positive
            fp = self.confusion_matrix[:, i].sum() - tp  # False Positive
            fn = self.confusion_matrix[i, :].sum() - tp  # False Negative

            union = tp + fp + fn
            if union > 0:
                iou = tp / union
                ious.append(iou)

        return np.mean(ious) if ious else 0.0

    def get_confusion_matrix(self):
        return self.confusion_matrix
