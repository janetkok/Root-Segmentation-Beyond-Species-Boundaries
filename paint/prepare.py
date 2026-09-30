import glob
import os
import numpy as np
from PIL import Image
import shutil
import cv2
import re
from natsort import natsorted
def gradient_orientation_robust(img, ksize):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    orientation = np.arctan2(gy, gx)
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(magnitude, 75)
    orientation_norm = ((orientation + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    orientation_norm[magnitude < threshold] = 0
    orientation_norm = cv2.normalize(orientation_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return orientation_norm




im_list = sorted(glob.glob(os.path.join('/data/ezajk13/plant/painting/wheat_raw', '*.TIFF')))
i = 179

color_table = {
    (0, 0, 0): 0,
    (255, 156, 0): 1,
    (0, 0, 255): 2,
    (255, 0, 255): 3,
    (255, 0, 0): 4,
    (255, 255, 0): 5,
    (125, 60, 152): 6,
    (255, 3, 127): 7,
    (3, 252, 69): 8,
}


for j in range(len(im_list)):
    im = cv2.imread(im_list[j])
    print(im_list[j])
    cv2.imwrite('/data/ezajk13/plant/painting/images/{}.png'.format(i), im)
    im = cv2.imread(im_list[j], 0)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # color_path = im_list[j].replace('.TIFF', '.png')
    # height_cv2, width_cv2 = im.shape[:2]
    # im_arr = Image.open(color_path).convert('RGB')
    # im_arr = im_arr.resize((width_cv2, height_cv2), Image.NEAREST)
    # im_arr = np.array(im_arr)
    gradient3 = gradient_orientation_robust(im, 3)
    gradient5 = gradient_orientation_robust(im, 5)
    gradient7 = gradient_orientation_robust(im, 7)
    # # img = cv2.GaussianBlur(im, (15, 15), 0)

    # # im_75 = cv2.adaptiveThreshold(
    # #     img, 255,
    # #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # #     cv2.THRESH_BINARY_INV,
    # #     75, 1
    # # )

    # # im_55 = cv2.adaptiveThreshold(
    # #     img, 255,
    # #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # #     cv2.THRESH_BINARY_INV,
    # #     55, 1
    # # )

    # # im_35 = cv2.adaptiveThreshold(
    # #     img, 255,
    # #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # #     cv2.THRESH_BINARY_INV,
    # #     35, 1
    # # )

    # label_map = np.zeros((im_arr.shape[0], im_arr.shape[1]), dtype=np.uint8)
    # for color, label in color_table.items():
    #     mask = np.all(im_arr == color, axis=-1)
    #     label_map[mask] = label
    # np.save('/data/ezajk13/plant/wheat/labels/{}.npy'.format(i), label_map)
    cv2.imwrite('/data/ezajk13/plant/painting/images_gradient3/{}.png'.format(i), gradient3)
    # cv2.imwrite('/data/ezajk13/plant/unlabeled/images_gradient5/{}.png'.format(i), gradient5)
    cv2.imwrite('/data/ezajk13/plant/painting/images_gradient7/{}.png'.format(i), gradient7)
    # cv2.imwrite('/data/ezajk13/plant/unlabeled/images_35/{}.png'.format(i), im_35)
    # cv2.imwrite('/data/ezajk13/plant/unlabeled/images_75/{}.png'.format(i), im_75)
    # cv2.imwrite('/data/ezajk13/plant/unlabeled/images_55/{}.png'.format(i), im_55)

    i=i+1