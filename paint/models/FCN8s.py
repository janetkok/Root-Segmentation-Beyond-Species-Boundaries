import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import model_zoo
from torchvision import models
from torchvision.models.vgg import VGG


class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels
        self.bilinear = bilinear

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCN8s. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = pretrained_net.n_channels
        self.bilinear = bilinear

        self.n_classes = n_classes
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

        if bilinear:
            raise NotImplementedError(
                'Bilinear interpolation not implemented for FCNs. Only transposed convolutions are used. Set bilinear=False to fix this.')

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']

        score = self.bn1(self.relu(self.deconv1(x5)))
        score = score + x4
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class VGGNet(nn.Module):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False,
                 n_channels=3):
        super(VGGNet, self).__init__()

        self.n_channels = n_channels

        if pretrained:
            assert n_channels == 3, "pretrained model is trained on 3 channel images, please use n_channels=3 when using pretrained weights."

        self.features = make_layers(cfg[model], n_channels=n_channels)
        self.ranges = ranges[model]

        if pretrained:

            if model in vgg_model_urls:
                weights = vgg_model_urls[model]
                self.load_state_dict(model_zoo.load_url(weights), strict=False)
            else:
                raise ValueError("Invalid VGG model name")

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if remove_fc:
            self._remove_fc()

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def _remove_fc(self):
        """
            Removes the fully connected layer (classifier) from the VGGNet model.

            This is useful for models where the fully connected layers are not required,
            such as when adapting VGGNet for a Fully Convolutional Network (FCN).
            The method checks for the existence of the classifier (e.g., built from VGG16 classification model) attribute before
            attempting to delete it to prevent AttributeError.
            """
        if hasattr(self, 'classifier'):
            del self.classifier

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        return output


ranges = {
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
}

cfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

vgg_model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


def make_layers(cfg, batch_norm=False, n_channels=3):
    layers = []
    # in_channels = 3
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def fcn8s(num_classes):
    vgg_model = VGGNet(requires_grad=True, model='vgg16', pretrained=True, n_channels=3)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_classes=num_classes, bilinear=False)
    return fcn_model