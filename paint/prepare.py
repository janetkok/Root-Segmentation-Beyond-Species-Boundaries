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




im_list = sorted(glob.glob(os.path.join('/data/ezajk13/plant/multiclass/images', '*.png')))
i = 0

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

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    gradient3 = gradient_orientation_robust(im, 3)
    gradient7 = gradient_orientation_robust(im, 7)


    cv2.imwrite('/data/ezajk13/plant/wheat/images_gradient3/{}.png'.format(i), gradient3)
    cv2.imwrite('/data/ezajk13/plant/wheat/images_gradient7/{}.png'.format(i), gradient7)


    i=i+1