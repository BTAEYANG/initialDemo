import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    def forward(self, f_t, f_s):

        s_spatial_relation, s_spatial_pearson = self.stage_spatial_pearson(f_s)
        # s_channel_relation, s_channel_pearson = self.stage_channel_pearson(f_s)
        s_channel_relation = self.stage_channel_pearson(f_s)
        # s_sample_relation, s_sample_pearson = self.stage_sample_pearson(f_s)
        s_sample_relation = self.stage_sample_pearson(f_s)

        with torch.no_grad():
            t_spatial_relation, t_spatial_pearson = self.stage_spatial_pearson(f_t)
            # t_channel_relation, t_channel_pearson = self.stage_channel_pearson(f_t)
            t_channel_relation = self.stage_channel_pearson(f_t)
            # t_sample_relation, t_sample_pearson = self.stage_sample_pearson(f_t)
            t_sample_relation = self.stage_sample_pearson(f_t)

        return t_spatial_pearson, s_spatial_pearson, t_spatial_relation, s_spatial_relation, t_channel_relation, s_channel_relation, t_sample_relation, s_sample_relation

    @staticmethod
    def stage_spatial_pearson(f):

        temp_spatial = []
        for i in range(len(f)):
            temp_spatial.append(f[i].mean(dim=1, keepdim=False).view(f[i].shape[0], -1).unsqueeze(1))

        spatial_matrix_list = []
        for j in range(len(temp_spatial) - 1):
            spatial_matrix_list.append(
                (torch.bmm(temp_spatial[j].transpose(1, 2), temp_spatial[j + 1])).mean(dim=0, keepdim=False))

        pearson_list = []
        for m in spatial_matrix_list:
            if m.shape[0] == m.shape[1]:
                pearson_list.append(torch.corrcoef(m))
            else:
                pearson_list.append(torch.corrcoef(m))
                pearson_list.append(torch.corrcoef(m.t()))

        return spatial_matrix_list, pearson_list

    @staticmethod
    def stage_channel_pearson(f):

        temp_channel = []

        for i in range(len(f)):
            temp_channel.append(f[i].mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False))

        channel_matrix_list = []
        for j in range(len(temp_channel) - 1):
            channel_matrix_list.append(torch.mm(temp_channel[j].transpose(0, 1), temp_channel[j + 1]))

        return channel_matrix_list

    @staticmethod
    def stage_sample_pearson(f):

        temp_sample = []

        for i in range(len(f)):
            temp_sample.append(torch.stack(
                torch.split((f[i].mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)),
                            split_size_or_sections=1, dim=0)))

        sample_matrix_list = []
        for j in range(len(temp_sample) - 1):
            sample_matrix_list.append(torch.mm(temp_sample[j], temp_sample[j + 1].transpose(0, 1)))

        return sample_matrix_list


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

    def forward(self, t_spatial_pearson, s_spatial_pearson, t_spatial_relation, s_spatial_relation, t_channel_relation, s_channel_relation, t_sample_relation, s_sample_relation):

        spatial_pearson_loss = sum(self.loss(i, j) for i, j in zip(t_spatial_pearson, s_spatial_pearson))

        spatial_relation_loss = sum(self.loss(i, j) for i, j in zip(t_spatial_relation, s_spatial_relation))

        channel_relation_loss = sum(self.loss(i, j) for i, j in zip(t_channel_relation, s_channel_relation))

        sample_relation_loss = sum(self.loss(i, j) for i, j in zip(t_sample_relation, s_sample_relation))

        loss = [spatial_pearson_loss, spatial_relation_loss, channel_relation_loss, sample_relation_loss]
        factor = F.softmax(torch.Tensor(loss), dim=-1)

        loss = sorted(loss)
        factor = sorted(factor.tolist(), reverse=True)

        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t


if __name__ == '__main__':
    pass
    # x = torch.randn(64, 3, 32, 32)
    #
    # b, _, _, _ = x.shape
    #
    # s_net = resnet8x4(num_classes=100)
    #
    # t_net = resnet32x4(num_classes=100)
    #
    # s_feats, s_logit = s_net(x, is_feat=True, preact=False)
    # t_feats, t_logit = t_net(x, is_feat=True, preact=False)
    #
    # f_s = s_feats[:-1]
    #
    # with torch.no_grad():
    #     s_spatial_relation, s_spatial_pearson = SKD.stage_spatial_pearson(f_s)
    #     s_sample_relation, s_sample_pearson = SKD.stage_channel_pearson(f_s)
