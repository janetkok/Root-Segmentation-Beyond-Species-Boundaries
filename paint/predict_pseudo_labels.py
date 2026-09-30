#!/usr/bin/env python3
"""Generate pseudo labels using a finetuned 10-class UNet.

Reads image list, computes gradient features on-the-fly (same as test.py),
runs inference, and saves predictions as .npy and/or palette PNG files.
"""
import os
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from models.unet import UnetResnet34

# RGB color per class index (0-9)
COLOR_TABLE = np.array([
    [0, 0, 0],        # 0: Background
    [255, 156, 0],    # 1: Cortex
    [0, 0, 255],      # 2: Metaxylem
    [255, 0, 255],    # 3: Stele Tissue
    [255, 0, 0],      # 4: Aerenchyma
    [255, 255, 0],    # 5: Vascular Bundle
    [125, 60, 152],   # 6: Sclerenchyma
    [255, 3, 127],    # 7: Epidermis
    [3, 252, 69],     # 8: Endodermis
    [0, 255, 255],    # 9: New class (HMR)
], dtype=np.uint8)


def build_palette(color_table):
    """Build a flat 768-byte palette for PIL 'P' mode from an Nx3 color table."""
    palette = np.zeros(768, dtype=np.uint8)
    for i, rgb in enumerate(color_table):
        palette[i * 3: i * 3 + 3] = rgb
    return palette.tolist()


def gradient_orientation_robust(img, ksize):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    orientation = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    threshold = np.percentile(magnitude, 75)
    orientation_norm = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    orientation_norm[magnitude < threshold] = 0
    orientation_norm = cv2.normalize(orientation_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return orientation_norm


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels with finetuned model')
    parser.add_argument('--model-path', required=True, help='Path to finetuned .pth')
    parser.add_argument('--data-dir', default='/data/ezajk13/plant/painting',
                        help='Root dir containing images/ and images_binary/')
    parser.add_argument('--file-list', default='/home/ezajk13/plant_seg/paint/train_appwithout_new.txt',
                        help='Text file with one filename per line')
    parser.add_argument('--output-dir', default='/data/ezajk13/plant/painting/pseudo_labels_hmr',
                        help='Where to save .npy pseudo labels')
    parser.add_argument('--num-cls', type=int, default=10)
    parser.add_argument('--resize', type=int, default=1024,
                        help='Resize for inference (same as val_resize in training)')
    parser.add_argument('--save-png', action='store_true', default=False,
                        help='Save palette PNGs instead of .npy files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnetResnet34(num_classes=args.num_cls)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    w_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.file_list) as f:
        filenames = [line.strip() for line in f if line.strip()]

    palette = build_palette(COLOR_TABLE) if args.save_png else None

    print(f'Generating pseudo labels for {len(filenames)} images...')

    for fname in tqdm(filenames):
        im_path = os.path.join(args.data_dir, 'images', fname)
        im = cv2.imread(im_path, 0)
        if im is None:
            print(f'Warning: could not read {im_path}, skipping')
            continue

        im_R = gradient_orientation_robust(im, 3)
        im_G = cv2.imread(os.path.join(args.data_dir, 'images_binary', fname), 0)
        im_B = gradient_orientation_robust(im, 7)

        if im_G is None:
            print(f'Warning: no binary mask for {fname}, skipping')
            continue

        img = np.stack([im_R, im_G, im_B], axis=2)
        inputs = w_transform(img).unsqueeze(0).to(device)

        B, C, H, W = inputs.shape
        inputs = torch.nn.functional.interpolate(
            inputs, size=(args.resize, args.resize), mode='bilinear'
        )

        with torch.no_grad():
            output = model(inputs)
            output = torch.nn.functional.interpolate(output, size=(H, W), mode='nearest')
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if args.save_png:
            pil_img = Image.fromarray(pred, mode='P')
            pil_img.putpalette(palette)
            pil_img.save(os.path.join(args.output_dir, fname))
        else:
            np.save(os.path.join(args.output_dir, fname.replace('.png', '.npy')), pred)

    print(f'Done. Saved pseudo labels to {args.output_dir}')


if __name__ == '__main__':
    main()
