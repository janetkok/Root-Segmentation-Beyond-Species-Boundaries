import torch
import numpy as np


def compute_iou(pred, target, num_classes):
    """
    Compute the IoU for each class.

    Args:
        pred (torch.Tensor): Prediction tensor of shape [B, H, W]
        target (torch.Tensor): Target tensor of shape [B, H, W]
        num_classes (int): Number of classes including background

    Returns:
        iou (torch.Tensor): IoU for each class of shape [num_classes]
    """
    # Convert to long type for compatibility with indices
    pred = pred.long()
    target = target.long()

    # Initialize the IoU list
    iou_list = torch.zeros(num_classes, dtype=torch.float)

    # Flatten the tensors to simplify the calculation
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        # Create masks for the current class
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        # Compute intersection and union
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            iou_list[cls] = float('nan')  # To handle division by zero if no instances of class in the batch
        else:
            iou_list[cls] = intersection / union

    return iou_list


def compute_miou(pred, target, num_classes):
    """
    Compute the mean IoU over all classes.

    Args:
        pred (torch.Tensor): Prediction tensor of shape [B, H, W]
        target (torch.Tensor): Target tensor of shape [B, H, W]
        num_classes (int): Number of classes including background

    Returns:
        miou (float): Mean IoU
    """
    # Ensure the tensors are on the same device
    device = pred.device
    target = target.to(device)

    # Initialize IoU sum and count
    iou_sum = torch.zeros(num_classes, dtype=torch.float, device=device)
    count = 0

    # Compute IoU for each image in the batch
    for i in range(pred.size(0)):
        pred_img = pred[i]
        target_img = target[i]
        iou = compute_iou(pred_img, target_img, num_classes)

        # Sum IoU values
        iou_sum += torch.tensor(iou, device=device)
        count += 1

    # Compute the mean IoU
    miou = iou_sum / count
    miou = torch.nanmean(miou)  # Handle cases with NaN values in IoU

    return miou.item()