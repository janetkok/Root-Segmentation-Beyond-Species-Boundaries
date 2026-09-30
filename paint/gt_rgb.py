import os.path

import numpy as np
import cv2


color_table = np.array([[0, 0, 0], [255, 156, 0], [0, 0, 255], [255, 0, 255],
                        [255, 0, 0], [255, 255, 0], [125, 60, 152], [255, 3, 127], [3, 252, 69]])

import glob
import os
im_list = glob.glob(os.path.join('painting/labels', '*.npy'))

for im in im_list:
    label = np.load(im)
    label = color_table[label].astype(np.uint8)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join('painting/gt_rgb', os.path.basename(im).replace('npy', 'png')), label)