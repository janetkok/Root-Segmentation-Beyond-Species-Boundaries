import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import cv2
import tqdm


im_list = glob.glob(os.path.join('painting/images', '*.png'))

for path in tqdm.tqdm(im_list):
    base = os.path.basename(path)#.replace('.TIFF', '.png')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary = cv2.imread(os.path.join('painting/images_binary', base.replace('.TIFF', '.png')))
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)

    gt = cv2.imread(os.path.join('painting/gt_rgb', base.replace('.TIFF', '.png')))
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    binary_blur = cv2.imread(os.path.join('painting/base', base))
    binary_blur = cv2.cvtColor(binary_blur, cv2.COLOR_BGR2RGB)

    binary_triple = cv2.imread(os.path.join('painting/ours', base))
    binary_triple = cv2.cvtColor(binary_triple, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))  # 1行3列，图像大小可调整

    # 显示每张图片
    axes[0].imshow(img)
    axes[0].axis('off')  # 去除坐标轴

    axes[1].imshow(binary_blur)
    axes[1].axis('off')
    axes[1].set_title("base")

    axes[2].imshow(binary_triple)
    axes[2].axis('off')
    axes[2].set_title("ours")

    axes[3].imshow(binary)
    axes[3].axis('off')

    axes[4].imshow(gt)
    axes[4].axis('off')
    axes[4].set_title("GT")


    plt.tight_layout()
    plt.savefig(os.path.join('painting/compare', base), bbox_inches='tight')