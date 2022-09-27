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
        bs, max_t_c = feat_t[-1].shape

        bs, max_s_c = feat_s[-1].shape

        feat_t[-1] = feat_t[-1].view(bs, max_t_c, 1, 1)
        feat_s[-1] = feat_s[-1].view(bs, max_s_c, 1, 1)

        max_c = max(max_t_c, max_s_c)

        self.lat_layer = nn.ModuleList([])
        self.resize_layer = nn.ModuleList([])
        base_c = [32, 64, 128, 256, 512]
        for index, value in enumerate(feat_t):
            _, c, h, w = value.shape
            self.resize_layer.append(add_conv(c, max(c, base_c[index]), 3, 1))
            self.lat_layer.append(add_conv(max(c, base_c[index]), max_c, 3, 1))

        # se reduction = 16
        c = int(max_c)
        red = int(max_c ** 0.5)
        self.se = SELayer(channel=c, reduction=red)

        # feature pyramid smooth
        self.smooth = add_conv(max_c, max_c, 3, 1)

        self.softmax_h = torch.nn.Softmax(dim=0)
        self.softmax_w = torch.nn.Softmax(dim=1)

        if torch.cuda.is_available():
            self.resize_layer.cuda()
            self.lat_layer.cuda()
            self.smooth.cuda()
            self.se.cuda()

    @staticmethod
    def up_sample_add(x, y):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f_t, f_s):

        bs, max_t_c = f_t[-1].shape
        bs, max_s_c = f_s[-1].shape

        f_t[-1] = f_t[-1].view(bs, max_t_c, 1, 1)
        f_s[-1] = f_s[-1].view(bs, max_s_c, 1, 1)

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
            if j == 0:
                t_p.append(self.lat_layer[len(resize_ft) - j - 1](ft))
            else:
                temp = self.lat_layer[len(resize_ft) - j - 1](ft)
                t_p.append(self.up_sample_add(t_p[t_p_i], temp))
                t_p_i = t_p_i + 1

        s_p = []
        s_p_i = 0
        for j, st in enumerate(resize_fs[::-1]):
            if j == 0:
                s_p.append(self.lat_layer[len(resize_fs) - j - 1](st))
            else:
                temp = self.lat_layer[len(resize_fs) - j - 1](st)
                s_p.append(self.up_sample_add(s_p[s_p_i], temp))
                s_p_i = s_p_i + 1

        # 在得到相加后的特征后，利用3×3卷积对生成的P1至P3再进行融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图
        p_t = [self.smooth(t) for t in t_p[::-1]]
        p_s = [self.smooth(s) for s in s_p[::-1]]

        # se_p_t = [self.se(t) for t in p_t]
        # se_p_s = [self.se(s) for s in p_s]

        t_dot_product_l = []
        t_dot_product_h_l = []
        t_dot_product_w_l = []
        t_pearson = []
        t_pearson_h = []
        t_pearson_w = []

        s_dot_product_l = []
        s_dot_product_h_l = []
        s_dot_product_w_l = []
        s_pearson = []
        s_pearson_h = []
        s_pearson_w = []


        for i, j, k, m in zip(f_t[:-1], p_t[:-1], f_s[:-1], p_s[:-1]):
            t_dot_product = (i.mean(dim=1, keepdim=False).view(i.shape[0], -1)) @ (
                j.mean(dim=1, keepdim=False).view(j.shape[0], -1).transpose(0, 1))

            t_dot_product_h = self.softmax_h(t_dot_product) / t_dot_product.shape[0]
            t_dot_product_w = self.softmax_w(t_dot_product) / t_dot_product.shape[1]

            t_dot_product_l.append(t_dot_product)
            t_dot_product_h_l.append(t_dot_product_h)
            t_dot_product_w_l.append(t_dot_product_w)

            t_pearson.append(torch.corrcoef(t_dot_product))
            # t_pearson_h.append(torch.corrcoef(t_dot_product_h))
            # t_pearson_w.append(torch.corrcoef(t_dot_product_w))

            s_dot_product = (k.mean(dim=1, keepdim=False).view(k.shape[0], -1)) @ (
                m.mean(dim=1, keepdim=False).view(m.shape[0], -1).transpose(0, 1))

            s_dot_product_h = self.softmax_h(s_dot_product) / s_dot_product.shape[0]
            s_dot_product_w = self.softmax_w(s_dot_product) / s_dot_product.shape[1]

            s_dot_product_l.append(s_dot_product)
            s_dot_product_h_l.append(s_dot_product_h)
            s_dot_product_w_l.append(s_dot_product_w)

            s_pearson.append(torch.corrcoef(s_dot_product))
            # s_pearson_h.append(torch.corrcoef(s_dot_product_h))
            # s_pearson_w.append(torch.corrcoef(s_dot_product_w))


        t_l = [torch.stack(t_dot_product_l), torch.stack(t_dot_product_h_l), torch.stack(t_dot_product_w_l), torch.stack(t_pearson)]
        s_l = [torch.stack(s_dot_product_l), torch.stack(s_dot_product_h_l), torch.stack(s_dot_product_w_l), torch.stack(s_pearson)]

        t_tensor = torch.stack(t_l)
        s_tensor = torch.stack(s_l)

        return t_tensor, s_tensor


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

    def forward(self, t_tensor, s_tensor):

        loss = sum(self.loss(i, j) for i, j in zip(t_tensor, s_tensor))

        return loss


if __name__ == '__main__':
    pass
    data = torch.randn(2, 3, 32, 32)
    x = torch.randn(64, 3, 32, 32)

    b, _, _, _ = x.shape

    s_net = resnet8x4(num_classes=100)
    t_net = resnet32x4(num_classes=100)

    feat_t, _ = s_net(data, is_feat=True, preact=False, feat_preact=False)
    feat_s, _ = t_net(data, is_feat=True, preact=False, feat_preact=False)

    f_t, s_logit = s_net(x, is_feat=True, preact=False, feat_preact=False)
    f_s, t_logit = t_net(x, is_feat=True, preact=False, feat_preact=False)

    skd = SKD(feat_t, feat_s)

    with torch.no_grad():
        g_t, g_s, se_g_t, se_g_s = skd(f_t, f_s)
