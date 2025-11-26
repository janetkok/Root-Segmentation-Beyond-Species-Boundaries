import cv2
from torchvision import transforms
import torch.utils.data as data
import os
from glob import glob
import numpy as np
import random
import torchvision.transforms.functional as F
import torch
from PIL import Image


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gradient_orientation_robust(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(magnitude, 75)
    orientation_norm = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    orientation_norm[magnitude < threshold] = 0
    return orientation_norm


class Plant(data.Dataset):
    def __init__(self, root_path, method='train', crop_size=512):
        self.root_path = root_path

        if method not in ['train', 'val']:
            raise Exception("not implemented")
        self.method = method
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.im_list = glob(os.path.join(self.root_path, method, 'images/*.png'))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        img_R = cv2.imread(img_path.replace('images', 'images_R'), 0)
        img_G = cv2.imread(img_path.replace('images', 'images_G'), 0)
        img_B = cv2.imread(img_path.replace('images', 'images_B'), 0)
        img = np.stack([img_R, img_G, img_B], axis=2)
        if self.method == 'train':
            gd_path = img_path.replace('images', 'masks')
            labels = Image.open(gd_path).convert('L')
            labels = (np.array(labels) >= 0.5).astype(np.uint8)
            img, labels = self.resize_crop_transform(img, labels)
        else:
            gd_path = img_path.replace('images', 'labels').replace('png', 'npy')
            labels = np.load(gd_path)
            labels[labels > 1] = 1
        return self.transform(img), torch.from_numpy(labels.copy()).long().unsqueeze(0)

    def resize_crop_transform(self, img, labels):
        if random.random() < 0.5:
            W, H = self.crop_size, self.crop_size
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
            labels = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            H, W, C = img.shape
            i, j, h, w = random_crop(H, W, self.crop_size, self.crop_size)
            labels = labels[i: (i + h), j: (j + w)]
            img = img[i: (i + h), j: (j + w)]

        if random.random() > 0.5:
            img = np.fliplr(img)
            labels = np.fliplr(labels)
        return img.copy(), labels