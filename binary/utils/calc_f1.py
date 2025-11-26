import torch


def calculate_multiclass_f1(pred, target, num_classes):
    """
    Calculate the F1 score of multiclass segmentation predictions.

    Args:
    - pred (torch.Tensor): Predicted segmentation masks with shape [B, C, H, W]
    - target (torch.Tensor): Ground truth segmentation masks with shape [B, H, W]
    - num_classes (int): Number of classes

    Returns:
    - float: Average F1 score of the predictions
    """
    # 确保 pred 和 target 的形状相同


    f1_scores = []

    for cls in range(num_classes):
        # 创建布尔掩码
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # 计算 TP, FP, FN
        TP = (pred_mask & target_mask).sum().item()
        FP = (pred_mask & ~target_mask).sum().item()
        FN = (~pred_mask & target_mask).sum().item()

        # 计算 Precision 和 Recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 计算 F1 分数
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        f1_scores.append(f1)

    # 计算平均 F1 分数
    average_f1 = sum(f1_scores) / len(f1_scores)

    return average_f1