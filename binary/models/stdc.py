import torch
import torch.nn as nn
import torch.nn.functional as F


# Regular convolution with kernel size 3x3
def conv3x3(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


# Regular convolution with kernel size 1x1, a.k.a. point-wise convolution
def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def channel_shuffle(x, groups=2):
    # Codes are borrowed from
    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# Depth-wise seperable convolution with batchnorm and activation
class DSConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, act_type='relu', **kwargs):
        super(DSConvBNAct, self).__init__(
            DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, **kwargs),
            PWConvBNAct(in_channels, out_channels, act_type, **kwargs)
        )


# Depth-wise convolution -> batchnorm -> activation
class DWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(DWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Point-wise convolution -> batchnorm -> activation
class PWConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu', bias=True, **kwargs):
        super(PWConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


# Transposed /de- convolution -> batchnorm -> activation
class DeConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None,
                 padding=None, act_type='relu', **kwargs):
        super(DeConvBNAct, self).__init__()
        if kernel_size is None:
            kernel_size = 2 * scale_factor - 1
        if padding is None:
            padding = (kernel_size - 1) // 2
        output_padding = scale_factor - 1
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=scale_factor, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )

    def forward(self, x):
        return self.up_conv(x)


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, pool_sizes=[1, 2, 4, 6], bias=False):
        super(PyramidPoolingModule, self).__init__()
        assert len(pool_sizes) == 4, 'Length of pool size should be 4.\n'
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, pool_sizes[0])
        self.stage2 = self._make_stage(in_channels, hid_channels, pool_sizes[1])
        self.stage3 = self._make_stage(in_channels, hid_channels, pool_sizes[2])
        self.stage4 = self._make_stage(in_channels, hid_channels, pool_sizes[3])
        self.conv = PWConvBNAct(2 * in_channels, out_channels, act_type=act_type, bias=bias)

    def _make_stage(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            conv1x1(in_channels, out_channels)
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.stage1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.stage2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.stage3(x), size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.stage4(x), size, mode='bilinear', align_corners=True)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=128):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            conv1x1(hid_channels, num_class)
        )


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super(AttentionRefinementModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNAct(channels, channels, 1, act_type='sigmoid')

    def forward(self, x):
        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv(x_pool)
        x = x * x_pool

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
                                conv1x1(out_channels, out_channels),
                                nn.ReLU(),
                                conv1x1(out_channels, out_channels),
                                nn.Sigmoid(),
                            )

    def forward(self, x_low, x_high):
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv1(x)

        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv2(x_pool)

        x_pool = x * x_pool
        x = x + x_pool

        return x


class STDC(nn.Module):
    def __init__(self, num_class=1, n_channel=3, encoder_type='stdc1', use_detail_head=False, use_aux=False,
                 act_type='relu'):
        super(STDC, self).__init__()
        repeat_times_hub = {'stdc1': [1, 1, 1], 'stdc2': [3, 4, 2]}
        if encoder_type not in repeat_times_hub.keys():
            raise ValueError('Unsupported encoder type.\n')
        repeat_times = repeat_times_hub[encoder_type]
        assert not use_detail_head * use_aux, 'Currently only support either aux-head or detail head.\n'
        self.use_detail_head = use_detail_head
        self.use_aux = use_aux

        self.stage1 = ConvBNAct(n_channel, 32, 3, 2)
        self.stage2 = ConvBNAct(32, 64, 3, 2)
        self.stage3 = self._make_stage(64, 256, repeat_times[0], act_type)
        self.stage4 = self._make_stage(256, 512, repeat_times[1], act_type)
        self.stage5 = self._make_stage(512, 1024, repeat_times[2], act_type)

        if use_aux:
            self.aux_head3 = SegHead(256, num_class, act_type)
            self.aux_head4 = SegHead(512, num_class, act_type)
            self.aux_head5 = SegHead(1024, num_class, act_type)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm4 = AttentionRefinementModule(512)
        self.arm5 = AttentionRefinementModule(1024)
        self.conv4 = conv1x1(512, 256)
        self.conv5 = conv1x1(1024, 256)

        self.ffm = FeatureFusionModule(256 + 256, 128, act_type)

        self.seg_head = SegHead(128, num_class, act_type)
        if use_detail_head:
            self.detail_head = SegHead(256, 1, act_type)
            self.detail_conv = conv1x1(3, 1)

    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]

        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x, is_training=False):
        size = x.size()[2:]

        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        if self.use_aux:
            aux3 = self.aux_head3(x3)

        x4 = self.stage4(x3)
        if self.use_aux:
            aux4 = self.aux_head4(x4)

        x5 = self.stage5(x4)
        if self.use_aux:
            aux5 = self.aux_head5(x5)

        x5_pool = self.pool(x5)
        x5 = x5_pool + self.arm5(x5)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)

        x4 = self.arm4(x4)
        x4 = self.conv4(x4)
        x4 += x5
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.ffm(x4, x3)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        if self.use_detail_head:
            x_detail = self.detail_head(x3)
            return x, x_detail
        elif self.use_aux:
            return x, (aux3, aux4, aux5)
        else:
            return x


class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(STDCModule, self).__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        if stride not in [1, 2]:
            raise ValueError(f'Unsupported stride: {stride}\n')

        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels // 2, 1)
        self.block2 = ConvBNAct(out_channels // 2, out_channels // 4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels // 4, out_channels // 8, 3)
        self.block4 = ConvBNAct(out_channels // 8, out_channels // 8, 3)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        if self.stride == 2:
            x1 = self.pool(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x1, x2, x3, x4], dim=1)


class LaplacianConv(nn.Module):
    def __init__(self, device):
        super(LaplacianConv, self).__init__()
        self.laplacian_kernel = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]]).to(device)

    def forward(self, lbl):
        size = lbl.size()[2:]
        lbl_1x = F.conv2d(lbl, self.laplacian_kernel, stride=1, padding=1)
        lbl_2x = F.conv2d(lbl, self.laplacian_kernel, stride=2, padding=1)
        lbl_4x = F.conv2d(lbl, self.laplacian_kernel, stride=4, padding=1)

        lbl_2x = F.interpolate(lbl_2x, size, mode='nearest')
        lbl_4x = F.interpolate(lbl_4x, size, mode='nearest')

        lbl = torch.cat([lbl_1x, lbl_2x, lbl_4x], dim=1)

        return lbl

