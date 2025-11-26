import numpy as np

class DiceMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total_intersection = np.zeros(self.num_classes)
        self.total_pred = np.zeros(self.num_classes)
        self.total_target = np.zeros(self.num_classes)


    def update(self, pred, target):
        """
        pred: [H, W] 预测类别
        target: [H, W] 真实类别
        """

        for c in range(1, self.num_classes):
            pred_c = (pred == c).astype(float)
            target_c = (target == c).astype(float)

            intersection = (pred_c * target_c).sum()

            self.total_intersection[c] += intersection
            self.total_pred[c] += pred_c.sum()
            self.total_target[c] += target_c.sum()


    def compute_micro_dice(self):
        dice_scores = []
        for c in range(1, self.num_classes):
            if self.total_pred[c] + self.total_target[c] > 0:
                dice = (2.0 * self.total_intersection[c]) / \
                       (self.total_pred[c] + self.total_target[c])
                dice_scores.append(dice)
        return np.mean(dice_scores)