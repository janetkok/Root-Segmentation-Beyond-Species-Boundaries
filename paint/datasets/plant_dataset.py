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


class Plant(data.Dataset):
    def __init__(self, root_path, method='train', crop_size=512, val_path='val.txt', train_path='train.txt', test_path='test.txt',prob_full=0.5):
        self.root_path = root_path
        if method not in ['train', 'val', 'test']:
            raise Exception("not implemented")
        self.method = method
        self.crop_size = crop_size
        self.prob_full = prob_full
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.im_list = []
        if method == 'train':
            img_list_path = train_path
        elif method == 'val':
            img_list_path = val_path
        else:
            img_list_path = test_path
        try:
            with open(img_list_path) as f:
                for i in f:
                    self.im_list.append(os.path.join(self.root_path, 'images/{}'.format(i.strip())))
        except:
            raise Exception("please give right info")


    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_R = cv2.imread(img_path.replace('images', 'images_gradient3'), 0)
        img_G = cv2.imread(img_path.replace('images', 'images_binary'), 0)
        img_B = cv2.imread(img_path.replace('images', 'images_gradient7'), 0)
        img = np.stack([img_R, img_G, img_B], axis=2)

        gd_path = img_path.replace('images', 'labels').replace('png', 'npy')
        labels = np.load(gd_path)
        if self.method == 'train':
            img, labels = self.resize_crop_transform(img, labels)
        return self.transform(img), torch.from_numpy(labels.copy()).long().unsqueeze(0)

    def resize_crop_transform(self, img, labels):
        if random.random() < self.prob_full:
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

