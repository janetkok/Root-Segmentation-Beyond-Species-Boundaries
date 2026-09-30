#!/usr/bin/env python3
"""Finetune pretrained 9-class UNet on HMR images with new class 9 (10 classes total).

Expands the final classification layer and trains on a small labeled set.
Uses differential learning rates (lower for encoder, higher for decoder/head)
and class-weighted CE to help the new class overcome softmax competition.
"""
import sys
import os
import time
import random
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.unet import UnetResnet34
from losses.diceloss import DiceLoss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_crop(im_h, im_w, crop_h, crop_w):
    i = random.randint(0, im_h - crop_h)
    j = random.randint(0, im_w - crop_w)
    return i, j, crop_h, crop_w


class PlantHMR(Dataset):
    """Small dataset loader for the HMR finetuning images."""

    def __init__(self, data_dir, crop_size=768, prob_full=0.5):
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.prob_full = prob_full
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        label_dir = os.path.join(data_dir, 'labels')
        self.samples = sorted(
            f.replace('.npy', '')
            for f in os.listdir(label_dir) if f.endswith('.npy')
        )
        logging.info(f'Found {len(self.samples)} samples: {self.samples}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img_R = cv2.imread(os.path.join(self.data_dir, f'images_gradient3/{name}.png'), 0)
        img_G = cv2.imread(os.path.join(self.data_dir, f'images_binary/{name}.png'), 0)
        img_B = cv2.imread(os.path.join(self.data_dir, f'images_gradient7/{name}.png'), 0)
        img = np.stack([img_R, img_G, img_B], axis=2)
        labels = np.load(os.path.join(self.data_dir, f'labels/{name}.npy'))

        img, labels = self._augment(img, labels)
        return self.transform(img), torch.from_numpy(labels.copy()).long().unsqueeze(0)

    def _augment(self, img, labels):
        H, W, C = img.shape
        if random.random() < self.prob_full:
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
            labels = cv2.resize(labels, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        else:
            # Ensure image is large enough for cropping
            if H < self.crop_size or W < self.crop_size:
                scale = max(self.crop_size / H, self.crop_size / W) + 0.01
                img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
                labels = cv2.resize(labels, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)
                H, W, C = img.shape
            i, j, h, w = random_crop(H, W, self.crop_size, self.crop_size)
            img = img[i:i + h, j:j + w]
            labels = labels[i:i + h, j:j + w]

        if random.random() > 0.5:
            img = np.fliplr(img)
            labels = np.fliplr(labels)

        return img.copy(), labels


def expand_final_layer(state_dict, old_cls=9, new_cls=10):
    """Expand lastlayer.1.weight from [old_cls, 64, 3, 3] to [new_cls, 64, 3, 3].

    Preserves weights for existing classes; Kaiming-initializes new class channels.
    """
    key = 'lastlayer.1.weight'
    old_w = state_dict[key]  # [9, 64, 3, 3]
    new_w = torch.zeros(new_cls, *old_w.shape[1:])
    new_w[:old_cls] = old_w
    nn.init.kaiming_normal_(new_w[old_cls:])
    state_dict[key] = new_w
    logging.info(f'Expanded {key}: {list(old_w.shape)} -> {list(new_w.shape)}')
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Finetune UNet for HMR (9->10 classes)')
    parser.add_argument('--data-dir', default='/data/ezajk13/plant/painting_hmr')
    parser.add_argument('--pretrained', default='/home/ezajk13/plant_seg/paint/logs/paint.pth')
    parser.add_argument('--save-dir', default='/home/ezajk13/plant_seg/paint/logs/finetune_hmr')
    parser.add_argument('--num-cls', type=int, default=10)
    parser.add_argument('--old-cls', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                        help='Lower LR for encoder to prevent catastrophic forgetting')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--crop-size', type=int, default=768)
    parser.add_argument('--ce-weight', type=float, default=0.1)
    parser.add_argument('--new-cls-weight', type=float, default=5.0,
                        help='CE weight multiplier for new class(es) to overcome softmax competition')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--freeze-encoder', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'finetune.log')),
            logging.StreamHandler()
        ]
    )
    logging.info(f'Args: {args}')

    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Model: build with 10 classes, load expanded pretrained weights ----
    model = UnetResnet34(num_classes=args.num_cls)
    pretrained = torch.load(args.pretrained, map_location='cpu')
    pretrained = expand_final_layer(pretrained, args.old_cls, args.num_cls)
    model.load_state_dict(pretrained)
    logging.info(f'Loaded pretrained weights, expanded {args.old_cls} -> {args.num_cls} classes')

    # ---- Freeze or use differential LR for encoder ----
    encoder_prefixes = ('firstlayer', 'encoder', 'maxpool')
    if args.freeze_encoder:
        frozen_count = 0
        for name, param in model.named_parameters():
            if name.startswith(encoder_prefixes):
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f'Froze {frozen_count} encoder param groups.  Trainable: {trainable:,} / {total:,}')

    model.to(device)

    # ---- Data ----
    dataset = PlantHMR(args.data_dir, crop_size=args.crop_size, prob_full=0.5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # ---- Loss: class-weighted CE to boost new class ----
    ce_weights = torch.ones(args.num_cls, device=device)
    for c in range(args.old_cls, args.num_cls):
        ce_weights[c] = args.new_cls_weight
    logging.info(f'CE class weights: {ce_weights.tolist()}')

    dice_loss_fn = DiceLoss(args.num_cls).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights).to(device)

    # ---- Optimizer: differential LR (encoder=low, decoder/head=high) ----
    if args.freeze_encoder:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        encoder_params, decoder_params = [], []
        for name, param in model.named_parameters():
            if name.startswith(encoder_prefixes):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': decoder_params, 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        logging.info(f'Differential LR: encoder={args.encoder_lr}, decoder/head={args.lr}')

    # ---- Training loop (no validation — only 3 images) ----
    for epoch in range(args.max_epoch):
        model.train()
        epoch_dice, epoch_ce, n = 0.0, 0.0, 0
        t0 = time.time()

        for img, targets in loader:
            img = img.to(device)
            targets = targets.to(device)

            outputs = model(img)
            dice = dice_loss_fn(outputs, targets, softmax=True)
            ce = ce_loss_fn(outputs, targets.squeeze(1))
            loss = dice + args.ce_weight * ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_dice += dice.item()
            epoch_ce += ce.item()
            n += 1

        logging.info(
            f'Epoch {epoch:>3d}/{args.max_epoch - 1} | '
            f'Dice: {epoch_dice / n:.4f} | CE: {epoch_ce / n:.4f} | '
            f'{time.time() - t0:.1f}s'
        )

        if (epoch + 1) % args.save_every == 0 or epoch == args.max_epoch - 1:
            path = os.path.join(args.save_dir, f'finetune_epoch{epoch}.pth')
            torch.save(model.state_dict(), path)
            logging.info(f'Saved: {path}')


if __name__ == '__main__':
    main()
