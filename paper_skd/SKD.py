import torch
from torch import nn


def add_conv(in_ch, out_ch, k_size, stride, leaky=True, bn=True):
    conv_module = nn.Sequential()
    pad_size = (k_size - 1) // 2
    conv_module.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                             out_channels=out_ch, kernel_size=k_size, stride=stride,
                                             padding=pad_size, bias=False))
    if bn:
        conv_module.add_module('batch_norm', nn.BatchNorm2d(out_ch))

    if leaky:
        conv_module.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        conv_module.add_module('relu6', nn.ReLU6(inplace=True))

    return conv_module


class SKD(nn.Module):

    def __init__(self, feat_t, feat_s):
        super(SKD, self).__init__()

        self.stage = len(feat_t)

        self.conv = add_conv(self.stage, self.stage, 3, 1)

    @staticmethod
    def _relation_dist(f, eps=1e-12, squared=True):
        f_square = f.pow(2).sum(dim=1)
        f_matrix = f @ f.t()
        relation_matrix = (f_square.unsqueeze(1) + f_square.unsqueeze(0) - 2 * f_matrix).clamp(min=eps)
        if squared:
            relation_matrix = relation_matrix.sqrt()
        relation_matrix = relation_matrix.clone()
        relation_matrix[range(len(f)), range((len(f)))] = 0
        return relation_matrix

    def forward(self, f_t, f_s):
        # initial feature channel
        b, _, _, _ = f_t[0].shape

        f_t = [i.view(i.shape[0], -1) for i in f_t]
        f_s = [i.view(i.shape[0], -1) for i in f_s]

        relation_t = torch.stack([self._relation_dist(i) for i in f_t]).view(1, self.stage, b, b)
        relation_s = torch.stack([self._relation_dist(i) for i in f_s]).view(1, self.stage, b, b)

        relation_t = self.conv(relation_t)
        relation_s = self.conv(relation_s)

        return relation_t, relation_s


class SKD_Loss(nn.Module):
    """ Stage Relation Distilling Loss"""

    def __init__(self):
        super(SKD_Loss, self).__init__()
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

    def forward(self, r_t, r_s):
        loss_r_s = self._spatial_mean_loss(r_t, r_s)
        loss_r_c = self._channel_mean_loss(r_t, r_s)

        loss = loss_r_s + loss_r_c

        return loss
