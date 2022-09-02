import math

import torch
from torch import nn
import torch.nn.functional as F

from paper.SELayer import SELayer
from paper.ScaledDotProductAttention import ScaledDotProductAttention


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

        # initial feature channel
        b_0, c_0, h_0, w_0 = feat_t[0].shape
        b_1, c_1, h_1, w_1 = feat_t[1].shape
        b_2, c_2, h_2, w_2 = feat_t[2].shape
        b_3, c_3, h_3, w_3 = feat_t[3].shape

        # attention
        self.self_attention_0 = ScaledDotProductAttention(d_model=h_0 * h_0, d_k=h_0 * h_0, d_v=h_0 * h_0, h=8)
        self.se_attention_0 = SELayer(c_0, math.sqrt(c_0))

        self.self_attention_1 = ScaledDotProductAttention(d_model=h_1 * h_1, d_k=h_1 * h_1, d_v=h_1 * h_1, h=8)
        self.se_attention_1 = SELayer(c_1, math.sqrt(c_1))

        self.self_attention_2 = ScaledDotProductAttention(d_model=h_2 * h_2, d_k=h_2 * h_2, d_v=h_2 * h_2, h=8,
                                                          dropout=0)
        self.se_attention_2 = SELayer(c_2, math.sqrt(c_2))

        self.self_attention_3 = ScaledDotProductAttention(d_model=h_3 * h_3, d_k=h_3 * h_3, d_v=h_3 * h_3, h=8,
                                                          dropout=0)
        self.se_attention_3 = SELayer(c_3, math.sqrt(c_3))

        # expand channel
        resize_c_0 = max(c_0, 32)
        resize_c_1 = max(c_1, 64)
        resize_c_2 = max(c_2, 128)
        resize_c_3 = max(c_3, 256)

        self.lat_layer0 = add_conv(resize_c_0, 256, 3, 1)
        self.lat_layer1 = add_conv(resize_c_1, 256, 3, 1)
        self.lat_layer2 = add_conv(resize_c_2, 256, 3, 1)
        self.lat_layer3 = add_conv(resize_c_3, 256, 3, 1)

        # resnet change channel
        self.resize_0 = add_conv(c_0, resize_c_0, 3, 1)
        self.resize_1 = add_conv(c_1, resize_c_1, 3, 1)
        self.resize_2 = add_conv(c_2, resize_c_2, 3, 1)
        self.resize_3 = add_conv(c_3, resize_c_3, 3, 1)

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
        t_p3 = self.lat_layer3(f_t[3])  # 8*8*256
        t_p3 = self.self_attention_3(t_p3, t_p3, t_p3)
        t_p2 = self._upSample_add(t_p3, self.lat_layer2(f_t[2]))  # self.lat_layer2(f_t[2]): 16*16*128 -> 16*16*256  _upSample_add: t_p3 (8*8*256) -> (16*16*256)
        t_p2 = self.self_attention_2(t_p2, t_p2, t_p2)
        t_p1 = self._upSample_add(t_p2, self.lat_layer1(f_t[1]))  # 256 + (64 -> 256)
        t_p1 = self.self_attention_2(t_p1, t_p1, t_p1)
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
                    t[j].data *= 0
                    s[j].data *= 0
                    se_t[j].data *= 0
                    se_s[j].data *= 0

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
        loss_f_s = self._spatial_mean_loss(g_s, g_t)
        loss_f_c = self._channel_mean_loss(g_s, g_t)
        loss_se_f_s = self._spatial_mean_loss(se_g_s, se_g_t)
        loss_se_f_c = self._channel_mean_loss(se_g_s, se_g_t)

        loss = [loss_f_s, loss_f_c, loss_se_f_s, loss_se_f_c]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t
