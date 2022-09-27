import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4
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


class SKD(nn.Module):

    def __init__(self, feat_t, feat_s):
        super(SKD, self).__init__()

        # max feature channel
        _, max_c, h, w = feat_t[-1].shape

        self.lat_layer = []
        self.resize_layer = []
        self.base_c = [32, 64, 128, 256]
        for index, value in enumerate(feat_t):
            print(index, value.shape)
            _, c, h, w = value.shape
            self.resize_layer.append(add_conv(c, max(c, self.base_c[index]), 3, 1))
            self.lat_layer.append(add_conv(max(c, self.base_c[index]), max_c, 3, 1))

        # se reduction = 16
        self.se = SELayer(max_c, max_c ** 0.5)

        # feature pyramid smooth
        self.smooth = add_conv(max_c, max_c, 3, 1)

    @staticmethod
    def up_sample_add(x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f_t, f_s):

        # resnet110 resnet32: t0-16, t1-16, t2-32, t3-64  # resnet32x4 resnet8x4: t0-32, t1-64, t2-128, t3-256
        resize_ft = []
        resize_fs = []
        for i, v in enumerate(f_t):
            resize_ft.append(self.resize_layer[i](v))

        for i, v in enumerate(f_s):
            resize_fs.append(self.resize_layer[i](v))

        t_p = []
        t_p_i = 0
        for j, ft in enumerate(resize_ft[::-1]):
            if j == len(resize_ft):
                t_p.append(self.lat_layer[j](ft))
            else:
                temp = self.lat_layer[j](ft)
                t_p.append(self._upSample_add(t_p[t_p_i], temp))
                t_p_i = t_p_i + 1

        s_p = []
        s_p_i = 0
        for j, st in enumerate(resize_fs[::-1]):
            if j == len(resize_fs):
                s_p.append(self.lat_layer[j](st))
            else:
                temp = self.lat_layer[j](st)
                s_p.append(self._upSample_add(s_p[s_p_i], temp))
                s_p_i = s_p_i + 1

        # 在得到相加后的特征后，利用3×3卷积对生成的P1至P3再进行融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图
        g_t = [self.smooth(t) for t in t_p]
        g_s = [self.smooth(s) for s in s_p]

        se_g_t = [self.se(t) for t in g_t]
        se_g_s = [self.se(s) for s in g_s]

        return g_t, g_s, se_g_t, se_g_s

    # @staticmethod
    # def stage_similarity_relation(f):
    #
    #     temp = []
    #
    #     for i in range(len(f)):
    #         b, c, h, w = f[i].shape
    #         # temp.append(f[i].mean(dim=1, keepdim=False).view(b, -1))
    #
    #     similarity_relation_list = []
    #     for j in range(len(temp) - 1):
    #         similarity_relation_list.append(temp[j].transpose(0, 1) @ temp[j + 1])
    #
    #     return similarity_relation_list


class SKD_Loss(nn.Module):
    """ Stage Relation Distilling Loss"""

    def __init__(self, loss):
        super(SKD_Loss, self).__init__()

        loss_type = loss

        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

    def forward(self, g_t, g_s, se_g_t, se_g_s):

        g_loss = sum(self.loss(i, j) for i, j in zip(g_t, g_s))
        g_se_loss = sum(self.loss(i, j) for i, j in zip(se_g_t, se_g_s))

        # relation_loss = [b_loss, c_loss, w_loss, h_loss]
        # factor = F.softmax(torch.Tensor(relation_loss), dim=-1)

        # loss = sorted(relation_loss)
        # factor = sorted(factor.tolist(), reverse=True)

        relation_loss_t = g_loss + g_se_loss

        return relation_loss_t


if __name__ == '__main__':
    pass
    x = torch.randn(2, 3, 32, 32)

    b, _, _, _ = x.shape

    s_net = resnet8x4(num_classes=100)

    t_net = resnet32x4(num_classes=100)

    s_feats, s_logit = s_net(x, is_feat=True, preact=False, feat_preact=False)
    t_feats, t_logit = t_net(x, is_feat=True, preact=False, feat_preact=False)

    f_t = t_feats[:-1]
    f_s = s_feats[:-1]

    skd = SKD(f_t, f_s)

    with torch.no_grad():
        g_t, g_s, se_g_t, se_g_s = skd(f_t, f_s)
