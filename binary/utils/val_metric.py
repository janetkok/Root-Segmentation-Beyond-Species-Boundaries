import torch
import numpy as np
import torch.nn.functional as F

def compute_iou(pred, target, num_classes):
    """
    Compute the IoU for each class and the mean IoU.

    Parameters:
    pred (torch.Tensor): Predicted labels of shape [H, W]
    target (torch.Tensor): Ground truth labels of shape [H, W]
    num_classes (int): Number of classes

    Returns:
    float: Mean IoU score
    """
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            iou_per_class.append(float('nan'))  # If there is no ground truth, do not include in average
        else:
            iou_per_class.append(intersection / union)

    # Compute mean IoU, ignoring NaN values
    iou_per_class = np.array(iou_per_class)
    mean_iou = np.nanmean(iou_per_class)

    return mean_iou


def calculate_miou(pred, target, num_classes):
    """
    Calculate mIoU for segmentation results.

    Parameters:
    pred (torch.Tensor): Predicted segmentation map of shape [B, C, H, W]
    target (torch.Tensor): Ground truth segmentation map of shape [B, H, W]
    num_classes (int): Number of classes

    Returns:
    float: Mean IoU score
    """
    # Convert prediction to class indices
    pred = torch.argmax(pred, dim=1)  # [B, C, H, W] -> [B, H, W]

    # Ensure target is of type long
    target = target.long()

    # Initialize mean IoU
    mean_iou = 0.0

    # Iterate over the batch
    for i in range(pred.shape[0]):
        mean_iou += compute_iou(pred[i], target[i], num_classes)

    # Average over the batch
    mean_iou /= pred.shape[0]

    return mean_iou


def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    """
    Compute the Dice coefficient for a single class.
    y_true and y_pred are binary_blur tensors of the same shape.
    """
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + epsilon) / (torch.sum(y_true) + torch.sum(y_pred) + epsilon)


def multi_class_dice_coefficient(y_true, y_pred, num_classes, epsilon=1e-6):
    """
    Compute the multi-class Dice coefficient.
    y_true and y_pred are tensors of shape (batch_size, height, width) with integer class labels.
    num_classes is the number of classes.
    """
    y_true_one_hot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()
    y_pred_one_hot = F.one_hot(y_pred, num_classes).permute(0, 3, 1, 2).float()

    dice = []
    for i in range(num_classes):
        dice.append(dice_coefficient(y_true_one_hot[:, i, :, :], y_pred_one_hot[:, i, :, :], epsilon))

    return torch.mean(torch.tensor(dice))
