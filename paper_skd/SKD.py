import torch
from torch import nn

from models import resnet32x4, resnet8x4
from util.embedding_util import MLPEmbed, LinearEmbed
import torch.nn.functional as F


class SKD(nn.Module):

    def __init__(self, feat_t, feat_s, model_t):
        super(SKD, self).__init__()

        dim_in_l = []
        for i in range(len(feat_t) - 1):
            b_H, t_H = feat_t[i].shape[2], feat_t[i + 1].shape[2]
            if b_H > t_H:
                dim_in_l.append(int(t_H * t_H))
            else:
                dim_in_l.append(int(b_H * b_H))


        # mlp embedding
        self.embedding_l = nn.ModuleList([])
        for j in dim_in_l:
            # self.embedding_l.append(MLPEmbed(dim_in=j, dim_out=dim_in_l[-1]))
            self.embedding_l.append(LinearEmbed(dim_in=j, dim_out=256))

        if torch.cuda.is_available():
            self.embedding_l.cuda()

        self.fc_embedding = model_t.fc

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


    def forward(self, f_t, f_s):
        stage_list_t = self.compute_stage(f_t)
        stage_list_s = self.compute_stage(f_s)


        with torch.no_grad():
            for i in range(len(stage_list_t) - 1):
                stage_list_t[i] = self.embedding_l[i](stage_list_t[i])

        for i in range(len(stage_list_t)-1):
            stage_list_s[i] = self.embedding_l[i](stage_list_s[i])

        return stage_list_t, stage_list_s


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

    def forward(self, t_embedding, s_embedding):

        loss = []
        for t, s in zip(t_embedding, s_embedding):
            delta = torch.abs(t - s)
            loss += torch.mean((delta[:-1] * delta[1:]).sum(1))

        return loss


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
    # skd = SKD(feat_t[:-1], feat_s[:-1])
    #
    # with torch.no_grad():
    #     fsp_list_t, fsp_list_s = skd(f_t[:-1], f_s[:-1])
