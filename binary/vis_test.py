import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import cv2
import tqdm


im_list = glob.glob(os.path.join('task/sorghum_priority/Curated images', '*.TIFF'))

for path in tqdm.tqdm(im_list):
    base = os.path.basename(path).replace('.TIFF', '.png')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    binary_blur = cv2.imread(os.path.join('task/sorghum_priority/binary_blur', base))
    binary_blur = cv2.cvtColor(binary_blur, cv2.COLOR_BGR2RGB)

    binary_triple = cv2.imread(os.path.join('task/sorghum_priority/binary_triple', base))
    binary_triple = cv2.cvtColor(binary_triple, cv2.COLOR_BGR2RGB)

    binary_new = cv2.imread(os.path.join('task/sorghum_priority/binary_g3_35_g7', base))
    binary_new = cv2.cvtColor(binary_new, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1行3列，图像大小可调整

    # 显示每张图片
    axes[0].imshow(img)
    axes[0].axis('off')  # 去除坐标轴

    axes[1].imshow(binary_blur)
    axes[1].axis('off')

    axes[2].imshow(binary_triple)
    axes[2].axis('off')

    axes[3].imshow(binary_new)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('task/sorghum_priority/figures', base), bbox_inches='tight')