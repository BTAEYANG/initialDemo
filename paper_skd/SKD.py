import torch
from torch import nn
import torch.nn.functional as F


def add_conv(in_ch, out_ch, k_size, stride, leaky=False):
    conv_module = nn.Sequential()
    pad_size = (k_size - 1) // 2
    conv_module.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                             out_channels=out_ch, kernel_size=k_size, stride=stride,
                                             padding=pad_size, bias=False))

    conv_module.add_module('batch_norm', nn.BatchNorm2d(out_ch))

    if leaky:
        conv_module.add_module('leaky', nn.LeakyReLU(inplace=True))
    else:
        conv_module.add_module('relu6', nn.ReLU6(inplace=True))

    return conv_module


class SKD(nn.Module):

    def __init__(self, feat_t, feat_s, opt):
        super(SKD, self).__init__()

        self.stage = len(feat_t)

        self.bs = opt.batch_size

        self.conv = add_conv(self.stage, self.stage, 3, 1)

    @staticmethod
    def _relation_dist(f, eps=1e-12, squared=False):
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

        # sample distance loss
        with torch.no_grad():
            f_t = [self._relation_dist(i) for i in f_t]
            f_t = [i / (i[i > 0].mean()) for i in f_t]

        f_s = [self._relation_dist(i) for i in f_s]
        f_s = [i / (i[i > 0].mean()) for i in f_s]

        relation_t_d = torch.stack(f_t).view(1, self.stage, b, b)
        relation_s_d = torch.stack(f_s).view(1, self.stage, b, b)

        relation_t_d = self.conv(relation_t_d)
        relation_s_d = self.conv(relation_s_d)

        # sample angle loss
        with torch.no_grad():
            f_t = [i.view(i.shape[0], -1) for i in f_t]
            td = [i.unsqueeze(0) - i.unsqueeze(1) for i in f_t]
            t_angle = torch.stack([torch.bmm(F.normalize(i, p=2, dim=2), F.normalize(i, p=2, dim=2).transpose(1, 2)) for i in td]).transpose(0, 1)

        f_s = [i.view(i.shape[0], -1) for i in f_s]
        sd = [i.unsqueeze(0) - i.unsqueeze(1) for i in f_s]
        s_angle = torch.stack([torch.bmm(F.normalize(i, p=2, dim=2), F.normalize(i, p=2, dim=2).transpose(1, 2)) for i in sd]).transpose(0, 1)

        relation_t_a = self.conv(t_angle)
        relation_s_a = self.conv(s_angle)

        return relation_t_d, relation_s_d, relation_t_a, relation_s_a


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

    def forward(self, r_t_d, r_s_d, r_t_a, r_s_a):
        loss_d_s = self._spatial_mean_loss(r_t_d, r_s_d)
        loss_d_c = self._channel_mean_loss(r_t_d, r_s_d)

        loss_a_s = self._spatial_mean_loss(r_t_a, r_s_a)
        loss_a_c = self._channel_mean_loss(r_t_a, r_s_a)

        loss = loss_d_s + loss_d_c + loss_a_s + loss_a_c

        return loss


