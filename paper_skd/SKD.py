import argparse

import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, ShuffleV1, resnet8x4, wrn_40_1, wrn_40_2
from models.mobilenetv2 import MobileNetV2, mobile_half
from models.vgg import vgg8, vgg13
from util.embedding_util import MLPEmbed


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    @staticmethod
    def compute_stage(g):
        stage_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.mean(dim=1, keepdim=False)
            top = top.mean(dim=1, keepdim=False)
            stage_list.append(torch.bmm(bot, top.transpose(1, 2)).view(bot.shape[0], -1))
        return stage_list

    def forward(self, f_t, f_s, embed_s, embed_t, model_t, opt):
        stage_list_t = self.compute_stage(f_t)
        stage_list_s = self.compute_stage(f_s)

        stage_list_fc_t = []
        stage_list_fc_s = []

        for i in range(len(stage_list_t)):
            stage_list_t[i] = embed_t[i](stage_list_t[i])
            if opt.model_t.__contains__('vgg'):
                stage_list_fc_t.append(model_t.classifier(stage_list_t[i]))
            elif opt.model_t.__contains__('ResNet'):
                stage_list_fc_t.append(model_t.linear(stage_list_t[i]))
            else:
                stage_list_fc_t.append(model_t.fc(stage_list_t[i]))

        for i in range(len(stage_list_s)):
            stage_list_s[i] = embed_s[i](stage_list_s[i])
            if opt.model_t.__contains__('vgg'):
                stage_list_fc_s.append(model_t.classifier(stage_list_s[i]))
            elif opt.model_t.__contains__('ResNet'):
                stage_list_fc_s.append(model_t.linear(stage_list_s[i]))
            else:
                stage_list_fc_s.append(model_t.fc(stage_list_s[i]))

        t_tensor = torch.stack(stage_list_t)
        s_tensor = torch.stack(stage_list_s)

        t_fc_tensor = torch.stack(stage_list_fc_t)
        s_fc_tensor = torch.stack(stage_list_fc_s)

        return t_tensor, s_tensor, t_fc_tensor, s_fc_tensor


class SKD_Loss(nn.Module):
    """ Stage Relation Distilling Loss"""

    def __init__(self, loss_type):
        super(SKD_Loss, self).__init__()

        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss(reduction='mean', delta=1.0)
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

    def _embedding_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = s.mean(dim=1, keepdim=False)
            t = t.mean(dim=1, keepdim=False)
            loss += self.loss(s, t)
        return loss

    def _sample_mean_loss(self, x, y):
        loss = 0.
        for s, t in zip(x, y):
            s = s.mean(dim=0, keepdim=False)
            t = t.mean(dim=0, keepdim=False)
            loss += self.loss(s, t)
        return loss

    def _stage_mean_loss(self, x, y):
        loss = self.loss(x.mean(dim=0, keepdim=False), y.mean(dim=0, keepdim=False))
        return loss

    def forward(self, t_tensor, s_tensor, t_fc_tensor, s_fc_tensor, opt):

        loss_ten_base = self.loss(t_tensor, s_tensor)
        loss_ten_e = self._embedding_mean_loss(s_tensor, t_tensor)
        loss_ten_sa = self._sample_mean_loss(s_tensor, t_tensor)
        loss_ten_st = self._stage_mean_loss(s_tensor, t_tensor)

        loss_fc_base = self.loss(t_fc_tensor, s_fc_tensor)
        loss_fc_e = self._embedding_mean_loss(s_fc_tensor, t_fc_tensor)
        loss_fc_sa = self._sample_mean_loss(s_fc_tensor, t_fc_tensor)
        loss_fc_st = self._stage_mean_loss(s_fc_tensor, t_fc_tensor)

        loss = [loss_ten_base, loss_fc_base, loss_ten_e, loss_ten_sa, loss_ten_st, loss_fc_e, loss_fc_sa, loss_fc_st]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        if opt.reverse:
            loss.reverse()
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t


if __name__ == '__main__':
    pass
    # x = torch.randn(64, 3, 32, 32)
    #
    # b, _, _, _ = x.shape
    #
    # s_net = mobile_half(num_classes=100)
    # t_net = vgg13(num_classes=100)
    #
    # f_s, s_logit = s_net(x, is_feat=True, preact=False)
    # f_t, t_logit = t_net(x, is_feat=True, preact=False)
    #
    # skd = SKD()
    #
    # embed_s = nn.ModuleList([])
    # embed_t = nn.ModuleList([])
    # f_t = f_t[:-1]
    # f_s = f_s[:-1]
    # dim_in_t = []
    # dim_in_s = []
    # for i in range(len(f_t) - 1):
    #     b_H, t_H = f_t[i].shape[2], f_t[i + 1].shape[2]
    #     if b_H > t_H:
    #         dim_in_t.append(int(t_H * t_H))
    #     else:
    #         dim_in_t.append(int(b_H * b_H))
    #
    # for k in range(len(f_s) - 1):
    #     s_b_H, s_t_H = f_s[k].shape[2], f_s[k + 1].shape[2]
    #     if s_b_H > s_t_H:
    #         dim_in_s.append(int(s_t_H * s_t_H))
    #     else:
    #         dim_in_s.append(int(s_b_H * s_b_H))
    #
    # for t, s in zip(dim_in_t, dim_in_s):
    #     if f_s[-1].shape[1] == f_t[-1].shape[1]:
    #         embed_s.append(MLPEmbed(dim_in=t, dim_out=f_t[-1].shape[1]))
    #         embed_t.append(MLPEmbed(dim_in=t, dim_out=f_t[-1].shape[1]))
    #     else:
    #         embed_s.append(MLPEmbed(dim_in=s, dim_out=f_t[-1].shape[1]))
    #         embed_t.append(MLPEmbed(dim_in=t, dim_out=f_t[-1].shape[1]))
    #
    # with torch.no_grad():
    #     parser = argparse.ArgumentParser('argument for training')
    #     parser.add_argument('--model_t', type=str, default='vgg13', help='')
    #     parser.add_argument('--reverse', default='False', action='store_true', help='reverse loss factor')
    #     opt = parser.parse_args()
    #     t_tensor, s_tensor, t_fc_tensor, s_fc_tensor = skd(f_t, f_s, embed_s, embed_t, t_net, opt)
    #     loss = SKD_Loss('SmoothL1')
    #     loss_val = loss(t_tensor, s_tensor, t_fc_tensor, s_fc_tensor, opt)
