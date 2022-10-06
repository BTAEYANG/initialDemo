import torch
from torch import nn

from models import resnet32x4, resnet8x4
from util.embedding_util import MLPEmbed, LinearEmbed
import torch.nn.functional as F


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
            stage_list.append(torch.bmm(bot, top).view(bot.shape[0], -1))
        return stage_list


    def forward(self, f_t, f_s, embed_s, embed_t):
        stage_list_t = self.compute_stage(f_t)
        stage_list_s = self.compute_stage(f_s)

        for i in range(len(stage_list_t)):
            stage_list_t[i] = embed_t[i](stage_list_t[i])

        for i in range(len(stage_list_s)):
            stage_list_s[i] = embed_s[i](stage_list_s[i])

        t_tensor = torch.stack(stage_list_t)
        s_tensor = torch.stack(stage_list_s)

        return t_tensor, s_tensor


class SKD_Loss(nn.Module):
    """ Stage Relation Distilling Loss"""

    def __init__(self, loss_type):
        super(SKD_Loss, self).__init__()

        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss()
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

    def forward(self, t_tensor, s_tensor):

        # loss_l = [self.loss(t, s) for t, s in zip(t_embedding, s_embedding)]
        # factor = F.softmax(torch.Tensor(loss_l), dim=-1)
        # loss_t = sum(factor[index] * loss_l[index] for index, value in enumerate(loss_l))
        loss_t = self.loss(t_tensor, s_tensor)
        return loss_t


if __name__ == '__main__':
    pass
    # data = torch.randn(2, 3, 32, 32)
    # x = torch.randn(64, 3, 32, 32)
    #
    # b, _, _, _ = x.shape
    #
    # s_net = resnet8x4(num_classes=100)
    # t_net = resnet32x4(num_classes=100)
    #
    # feat_t, _ = s_net(data, is_feat=True, preact=False, feat_preact=False)
    # feat_s, _ = t_net(data, is_feat=True, preact=False, feat_preact=False)
    #
    # f_t, s_logit = s_net(x, is_feat=True, preact=False, feat_preact=False)
    # f_s, t_logit = t_net(x, is_feat=True, preact=False, feat_preact=False)
    #
    # skd = SKD(feat_t[:-1], feat_s[:-1], t_net)
    #
    # with torch.no_grad():
    #     fsp_list_t, fsp_list_s = skd(f_t[:-1], f_s[:-1])
    #     loss = SKD_Loss('SmoothL1')
    #     loss_val = loss(fsp_list_t, fsp_list_s)
