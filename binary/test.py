import glob
import os
from models.unet import UnetResnet34
import torch
from PIL import Image
from torchvision import transforms
import cv2

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


model = UnetResnet34(num_classes=2)
model.load_state_dict(torch.load('/home/ezajk13/plant_seg/binary/weights/binary.pth', map_location=torch.device('cpu')))
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
model = model.to(device)

data_list = sorted(glob.glob(os.path.join('/data/ezajk13/plant/wheat/raw', '*.TIFF')))
i=0
import numpy as np
import tqdm
#color_table = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 165, 0],
# [128, 0, 128], [0, 255, 255], [255, 0, 255], [255, 255, 0], [255, 192, 203]])
color_table = np.array([[0, 0, 0], [255, 255, 255]])

w_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
for im_path in tqdm.tqdm(data_list):
    img = cv2.imread(im_path, 0)
    im_G = cv2.GaussianBlur(img, (15, 15), 0)
    im_G = cv2.adaptiveThreshold(
        im_G, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 1
    )
    im_R = gradient_orientation_robust(img,3)
    #im_G = gradient_orientation_robust(img,5)
    im_B = gradient_orientation_robust(img, 7)
    img = np.stack([im_R, im_G, im_B], axis=2)
    # img = cv2.cvtColor(im_G, cv2.COLOR_GRAY2RGB)
    inputs = w_transform(img)
    inputs = inputs.unsqueeze(0).to(device)

    B, C, H, W = inputs.shape
    inputs = torch.nn.functional.interpolate(inputs, size=(1024, 1024), mode='bilinear')
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        output = torch.nn.functional.interpolate(output, size=(H, W), mode='nearest').argmax(dim=1).squeeze(0).detach().cpu().numpy()
        rgb_predict = color_table[output].astype(np.uint8)
        print(rgb_predict.shape)
    rgb_predict = cv2.cvtColor(rgb_predict, cv2.COLOR_RGB2BGR)
    cv2.imwrite('/data/ezajk13/plant/wheat/images_binary/{}.png'.format(i), rgb_predict)
    i+=1
    # plt.figure(figsize=(20, 5))
    # plt.subplot(121)
    # plt.imshow(im)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(color_table[output], cmap='gray')
    # # plt.title('Prediction'), plt.xticks([]), plt.yticks([])
    # # plt.subplot(133)
    # # plt.imshow(color_table[np.load(im_path.replace('png', 'npy').replace('Images', 'Labels'))])
    # # plt.title('Labels'), plt.xticks([]), plt.yticks([])
    #plt.savefig('image-sorghum/rgb/{}'.format(os.path.basename(im_path).replace('tif', 'png')), bbox_inches='tight',
    #            pad_inches=0)
    # plt.show()