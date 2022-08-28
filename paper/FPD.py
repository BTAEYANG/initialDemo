import torch
from torch import nn
import torch.nn.functional as F

from paper.SELayer import SELayer


def add_conv(in_ch, out_ch, k_size, stride, leaky=True):
    conv_module = nn.Sequential()
    pad_size = (k_size - 1) // 2
    conv_module.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                             out_channels=out_ch, kernel_size=k_size, stride=stride,
                                             padding=pad_size, bias=False))
    conv_module.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        conv_module.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        conv_module.add_module('relu6', nn.ReLU6(inplace=True))
    return conv_module


class FPD(nn.Module):
    """ Feature Pyramid Distilling """

    def __init__(self, feat_t, feat_s):
        super(FPD, self).__init__()

        # initial feature size
        initial_size_0 = feat_t[0].size()[1]
        initial_size_1 = feat_t[1].size()[1]
        initial_size_2 = feat_t[2].size()[1]
        initial_size_3 = feat_t[3].size()[1]

        # expand channel
        resize_0_channel = max(initial_size_0, 32)
        resize_1_channel = max(initial_size_1, 64)
        resize_2_channel = max(initial_size_2, 128)
        resize_3_channel = max(initial_size_3, 256)

        self.lat_layer0 = add_conv(resize_0_channel, 256, 3, 1)
        self.lat_layer1 = add_conv(resize_1_channel, 256, 3, 1)
        self.lat_layer2 = add_conv(resize_2_channel, 256, 3, 1)
        self.lat_layer3 = add_conv(resize_3_channel, 256, 3, 1)

        # resnet change channel
        self.resize_0 = add_conv(initial_size_0, resize_0_channel, 3, 1)
        self.resize_1 = add_conv(initial_size_1, resize_1_channel, 3, 1)
        self.resize_2 = add_conv(initial_size_2, resize_2_channel, 3, 1)
        self.resize_3 = add_conv(initial_size_3, resize_3_channel, 3, 1)

        # se reduction = 16
        self.se = SELayer(256, 16)

        # feature pyramid smooth
        self.smooth = add_conv(256, 256, 3, 1)

    @staticmethod
    def _upSample_add(x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f_t, f_s, error_index):
        # resnet110 resnet32: t0-16, t1-16, t2-32, t3-64
        f_t[0] = self.resize_0(f_t[0])
        f_t[1] = self.resize_1(f_t[1])
        f_t[2] = self.resize_2(f_t[2])
        f_t[3] = self.resize_3(f_t[3])

        f_s[0] = self.resize_0(f_s[0])
        f_s[1] = self.resize_1(f_s[1])
        f_s[2] = self.resize_2(f_s[2])
        f_s[3] = self.resize_3(f_s[3])

        # resnet32x4 resnet8x4: t0-32, t1-64, t2-128, t3-256

        # _upSample_add
        t_p3 = self.lat_layer3(f_t[3])  # 256
        t_p2 = self._upSample_add(t_p3, self.lat_layer2(f_t[2]))  # 256 + (128 -> 256)
        t_p1 = self._upSample_add(t_p2, self.lat_layer1(f_t[1]))  # 256 + (64 -> 256)
        t_p0 = self._upSample_add(t_p1, self.lat_layer0(f_t[0]))  # 256 + (32 -> 256)

        # f0-32, f1-64, f2-128, f3-256, f4-256 (out)
        # _upSample_add
        s_p3 = self.lat_layer3(f_s[3])  # 256
        s_p2 = self._upSample_add(s_p3, self.lat_layer2(f_s[2]))  # 256 + (128 -> 256)
        s_p1 = self._upSample_add(s_p2, self.lat_layer1(f_s[1]))  # 256 + (64 -> 256)
        s_p0 = self._upSample_add(s_p1, self.lat_layer0(f_s[0]))  # 256 + (32 -> 256)

        # t_p3 256*8*8  t_p2 256*16*16  t_p1 256*32*32  t_p0 256*32*32
        t_p = [t_p3, t_p2, t_p1, t_p0]
        s_p = [s_p3, s_p2, s_p1, s_p0]

        # 在得到相加后的特征后，利用3×3卷积对生成的P1至P3再进行融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图
        g_t = [self.smooth(t) for t in t_p]
        g_s = [self.smooth(s) for s in s_p]

        se_g_t = [self.se(t) for t in g_t]
        se_g_s = [self.se(s) for s in g_s]

        if len(error_index):
            for t, s, se_t, se_s in zip(g_t, g_s, se_g_t, se_g_s):
                for j in error_index:
                    t[j] *= 0
                    s[j] *= 0
                    se_t[j] *= 0
                    se_s[j] *= 0

        return g_t, g_s, se_g_t, se_g_s


class FPD_Loss(nn.Module):
    """ Feature Pyramid Distilling Loss"""

    def __init__(self):
        super(FPD_Loss, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mse = nn.MSELoss(reduction='mean')

    def _channel_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = self.avg_pool(s)
            t = self.avg_pool(t)
            loss += self.mse(s, t)
        return loss

    def _spatial_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = s.mean(dim=1, keepdim=False)
            t = t.mean(dim=1, keepdim=False)
            loss += self.mse(s, t)
        return loss

    def forward(self, g_t, g_s, se_g_t, se_g_s):
        loss_f = self._spatial_mean_loss(g_s, g_t)
        loss_se_f = self._channel_mean_loss(se_g_s, se_g_t)

        loss = [loss_f, loss_se_f]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t
