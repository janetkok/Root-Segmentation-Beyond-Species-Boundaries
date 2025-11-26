# import matplotlib.pyplot as plt
import glob
import os
from models.unet import UnetResnet34
import torch
from PIL import Image
from torchvision import transforms
import cv2
import argparse
import numpy as np
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


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model_path', default='',
                        help='model_path')
    parser.add_argument('--predict_path', default='',
                        help='predict path')
    parser.add_argument('--data_path', default='',
                        help='data path')
    parser.add_argument('--binary_path', default='images_binary',
                        help='binary path')
   

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
        
    model = UnetResnet34(num_classes=9)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model = model.to(device)

    data_list = glob.glob(os.path.join(args.data_path, '*.png'))


    color_table = np.array([[0, 0, 0], [255, 156, 0], [0, 0, 255], [255, 0, 255],
                            [255, 0, 0], [255, 255, 0], [125, 60, 152], [255, 3, 127], [3, 252, 69]])
    #color_table = np.array([[0, 0, 0], [255, 255, 255]])

    w_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    import tqdm

    os.makedirs(args.predict_path, exist_ok=True)
    for im_path in tqdm.tqdm(data_list):
        im = cv2.imread(im_path, 0)
        #img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_R = gradient_orientation_robust(im, 3)
        im_B = gradient_orientation_robust(im, 7)
        im_G = cv2.imread(im_path.replace('images', args.binary_path).replace('TIFF', 'png'), 0)
        img = np.stack([im_R, im_G, im_B], axis=2)
        inputs = w_transform(img)
        inputs = inputs.unsqueeze(0).to(device)

        B, C, H, W = inputs.shape
        inputs = torch.nn.functional.interpolate(inputs, size=(1024, 1024), mode='bilinear')
        model.eval()
        with torch.no_grad():
            output = model(inputs)
            #output_mask = torch.nn.functional.interpolate(output[:, 1:], size=(H, W), mode='nearest').argmax(dim=1).squeeze(0).detach().cpu().numpy() + 1
            output = torch.nn.functional.interpolate(output, size=(H, W), mode='nearest').argmax(dim=1).squeeze(0).detach().cpu().numpy()
            #out = output * (im_G < 0.5) + output_mask * (im_G >0.5)
            rgb_predict = color_table[output].astype(np.uint8)
            print(rgb_predict.shape)
        rgb_predict = cv2.cvtColor(rgb_predict, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.predict_path+'/{}'.format(os.path.basename(im_path)), rgb_predict)
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