import torch
import torch.nn as nn
import torch.nn.functional as F

from models import mobile_half, resnet32x4
from models.vgg import vgg13


class DWD(nn.Module):
    """Depth Wise Knowledge Distillation"""

    def __init__(self, f_t, f_s):
        super(DWD, self).__init__()

        t_depth_point_list = nn.ModuleList()
        s_depth_point_list = nn.ModuleList()

        for i in f_t:
            in_channel = i.shape[1]
            k_size = ((i.shape[2] // 2) - 1)
            if k_size < 1:
                t_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=in_channel),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            elif k_size < 3:
                t_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel, padding=1, stride=1),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            else:
                padding = (k_size - 1) // 2
                t_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=k_size, groups=in_channel, padding=padding, stride=1),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            t_depth_point_list.append(t_depth_point)

        for j in f_s:
            in_channel = j.shape[1]
            k_size = ((j.shape[2] // 2) - 1)
            if k_size < 1:
                s_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=1, groups=in_channel),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            elif k_size < 3:
                s_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel, padding=1, stride=1),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            else:
                padding = (k_size - 1) // 2
                s_depth_point = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=k_size, groups=in_channel, padding=padding, stride=1),
                    nn.Conv2d(in_channel, in_channel, kernel_size=1))
            s_depth_point_list.append(s_depth_point)

        self.t_depth_point_list = t_depth_point_list
        self.s_depth_point_list = s_depth_point_list

    def forward(self, f_t, f_s):

        f_t_l = [self.t_depth_point_list[index](t) for index, t in enumerate(f_t)]

        f_s_l = [self.s_depth_point_list[index](s) for index, s in enumerate(f_s)]

        f_t_res = [t + t_l for t, t_l in zip(f_t, f_t_l)]

        f_s_res = [s + s_l for s, s_l in zip(f_s, f_s_l)]

        return f_t, f_s, f_t_res, f_s_res


class DWD_Loss(nn.Module):
    """ Depth Wise Knowledge Distillation Loss"""

    def __init__(self, loss_type):
        super(DWD_Loss, self).__init__()

        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss(reduction='mean', delta=1.0)
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

    def _channel_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = self.avg_pool(s)
            t = self.avg_pool(t)
            loss += self.loss(s, t)
        return loss

    def _spatial_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = s.mean(dim=1, keepdim=False)
            t = t.mean(dim=1, keepdim=False)
            loss += self.loss(s, t)
        return loss

    def forward(self, f_t, f_s, f_t_res, f_s_res, opt):

        loss_base_s = self._spatial_mean_loss(f_t, f_s)
        loss_base_c = self._channel_mean_loss(f_t, f_s)
        loss_res_s = self._spatial_mean_loss(f_t_res, f_s_res)
        loss_res_c = self._channel_mean_loss(f_t_res, f_s_res)

        loss = [loss_base_s, loss_base_c, loss_res_s, loss_res_c]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        if opt.reverse:
            loss.reverse()
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t


if __name__ == '__main__':
    x = torch.randn(64, 3, 32, 32)

    s_net = mobile_half(num_classes=100)
    t_net = resnet32x4(num_classes=100)

    f_s, s_logit = s_net(x, is_feat=True, preact=False)
    f_t, t_logit = t_net(x, is_feat=True, preact=False)

    dwd = DWD(f_t[:-1], f_s[:-1])

    dwd(f_t[:-1], f_s[:-1])
