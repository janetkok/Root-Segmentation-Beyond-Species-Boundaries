import cv2
import numpy as np
import os
import glob
import tqdm
import shutil

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

k = 188
im_list = glob.glob(os.path.join('painting/images', '*.png'))
for im_path in tqdm.tqdm(im_list):
    im = cv2.imread(im_path, 0)
    height, width = im.shape[:2]
    img = cv2.GaussianBlur(im, (15, 15), 0)

    orientation_3 = gradient_orientation_robust(im, 3)
    orientation_7 = gradient_orientation_robust(im, 7)


    im_35 = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 1
    )

    cv2.imwrite('painting/images_R/{}'.format(os.path.basename(im_path)), orientation_3)
    cv2.imwrite('painting/images_B/{}'.format(os.path.basename(im_path)), orientation_7)
    cv2.imwrite('painting/images_G/{}'.format(os.path.basename(im_path)), im_35)


