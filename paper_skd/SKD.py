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
            stage_list.append(torch.bmm(bot, top.transpose(1, 2)).view(bot.shape[0], -1))
        return stage_list

    def forward(self, f_t, f_s, embed_s, embed_t, model_t):
        stage_list_t = self.compute_stage(f_t)
        stage_list_s = self.compute_stage(f_s)

        stage_list_fc_t = []
        stage_list_fc_s = []

        for i in range(len(stage_list_t)):
            stage_list_t[i] = embed_t[i](stage_list_t[i])
            stage_list_fc_t.append(model_t.fc(stage_list_t[i]))

        for i in range(len(stage_list_s)):
            stage_list_s[i] = embed_s[i](stage_list_s[i])
            stage_list_fc_s.append(model_t.fc(stage_list_s[i]))

        t_tensor = torch.stack(stage_list_t)
        s_tensor = torch.stack(stage_list_s)

        t_fc_tensor = torch.stack(stage_list_fc_t)
        s_fc_tensor = torch.stack(stage_list_fc_s)

        print(t_tensor.shape)
        print(s_tensor.shape)
        print(t_fc_tensor.shape)
        print(t_fc_tensor.shape)

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

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
            s = s.mean(dim=1, keepdim=False).mean(dim=2, keepdim=False)
            t = t.mean(dim=1, keepdim=False).mean(dim=2, keepdim=False)
            loss += self.loss(s, t)
        return loss

    def forward(self, t_tensor, s_tensor, t_fc_tensor, s_fc_tensor):

        loss_ten_base = self.loss(t_tensor, s_tensor)
        loss_fc_base = self.loss(t_fc_tensor, s_fc_tensor)

        loss_ten_s = self._spatial_mean_loss(s_tensor, t_tensor)
        loss_ten_c = self._channel_mean_loss(s_tensor, t_tensor)

        loss_f_s = self._spatial_mean_loss(s_fc_tensor, t_fc_tensor)
        loss_f_c = self._channel_mean_loss(s_fc_tensor, t_fc_tensor)

        loss = [loss_ten_base, loss_fc_base, loss_f_s, loss_f_c, loss_ten_s, loss_ten_c]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        loss.reverse()
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t


if __name__ == '__main__':
    pass
    # x = torch.randn(64, 3, 32, 32)
    #
    # b, _, _, _ = x.shape
    #
    # s_net = resnet8x4(num_classes=100)
    # t_net = resnet32x4(num_classes=100)
    #
    # f_t, s_logit = s_net(x, is_feat=True, preact=False, feat_preact=True)
    # f_s, t_logit = t_net(x, is_feat=True, preact=False, feat_preact=True)
    #
    # skd = SKD()
    #
    # embed_s = nn.ModuleList([])
    # embed_t = nn.ModuleList([])
    # f_t = f_t[:-1]
    # f_s = f_s[:-1]
    # dim_in_l = []
    # for i in range(len(f_t) - 1):
    #     b_H, t_H = f_t[i].shape[2], f_t[i + 1].shape[2]
    #     if b_H > t_H:
    #         dim_in_l.append(int(t_H * t_H))
    #     else:
    #         dim_in_l.append(int(b_H * b_H))
    #
    # for j in dim_in_l:
    #     embed_s.append(LinearEmbed(dim_in=j, dim_out=f_s[-1].shape[1]))
    #     embed_t.append(LinearEmbed(dim_in=j, dim_out=f_t[-1].shape[1]))
    #
    # with torch.no_grad():
    #     t_tensor, s_tensor, t_fc_tensor, s_fc_tensor = skd(f_t, f_s, embed_s, embed_t, t_net)
    #     loss = SKD_Loss('SmoothL1')
    #     loss_val = loss(t_tensor, s_tensor, t_fc_tensor, s_fc_tensor)
