import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    def forward(self, f_t, f_s):
        spatial_f_t, channel_f_t, sample_f_t = f_t, f_t, f_t
        spatial_f_s, channel_f_s, sample_f_s = f_s, f_s, f_s

        s_spatial_relation, s_spatial_pearson = self.stage_spatial_pearson(spatial_f_s)
        s_channel_relation, s_channel_pearson = self.stage_channel_pearson(channel_f_s)
        s_sample_relation, s_sample_pearson = self.stage_sample_pearson(sample_f_s)

        with torch.no_grad():
            t_spatial_relation, t_spatial_pearson = self.stage_spatial_pearson(spatial_f_t)
            t_channel_relation, t_channel_pearson = self.stage_channel_pearson(channel_f_t)
            t_sample_relation, t_sample_pearson = self.stage_sample_pearson(sample_f_t)

        return t_spatial_pearson, s_spatial_pearson, t_spatial_relation, s_spatial_relation, t_channel_pearson, s_channel_pearson, t_channel_relation, s_channel_relation, t_sample_pearson, s_sample_pearson, t_sample_relation, s_sample_relation

    @staticmethod
    def stage_spatial_pearson(f):

        for i in range(len(f)):
            f[i] = f[i].mean(dim=1, keepdim=False).view(f[i].shape[0], -1).unsqueeze(1)

        matrix_list = []
        for j in range(len(f) - 1):
            matrix_list.append((torch.bmm(f[j].transpose(1, 2), f[j + 1])).mean(dim=0, keepdim=False))

        pearson_list = []
        for m in matrix_list:
            if m.shape[0] == m.shape[1]:
                pearson_list.append(torch.corrcoef(m))
            else:
                pearson_list.append(torch.corrcoef(m))
                pearson_list.append(torch.corrcoef(m.t()))

        return matrix_list, pearson_list

    @staticmethod
    def stage_channel_pearson(f):
        for i in range(len(f)):
            print(f[i].shape)
            f[i] = f[i].mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).unsqueeze(0)
            print(f[i].shape)

        matrix_list = []
        for j in range(len(f) - 1):
            print(f[j].shape)
            print(f[j + 1].shape)
            matrix_list.append(torch.bmm(f[j].transpose(1, 2), f[j + 1]).mean(dim=0, keepdim=False))

        pearson_list = []
        for m in matrix_list:
            pearson_list.append(torch.corrcoef(m))

        return matrix_list, pearson_list

    @staticmethod
    def stage_sample_pearson(f):
        for i in range(len(f)):
            f[i] = torch.stack(
                torch.split((f[i].mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)),
                            split_size_or_sections=1, dim=0)).unsqueeze(0)

        matrix_list = []
        for j in range(len(f) - 1):
            matrix_list.append(torch.bmm(f[j], f[j + 1].transpose(1, 2)).mean(dim=0, keepdim=False))

        pearson_list = []
        for m in matrix_list:
            pearson_list.append(torch.corrcoef(m))

        return matrix_list, pearson_list


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

    def forward(self, t_spatial_pearson, s_spatial_pearson, t_spatial_relation, s_spatial_relation, t_channel_pearson,
                s_channel_pearson, t_channel_relation, s_channel_relation, t_sample_pearson, s_sample_pearson,
                t_sample_relation, s_sample_relation):

        spatial_pearson_loss = sum(self.loss(i, j) for i, j in zip(t_spatial_pearson, s_spatial_pearson))
        spatial_relation_loss = sum(self.loss(i, j) for i, j in zip(t_spatial_relation, s_spatial_relation))

        channel_pearson_loss = sum(self.loss(i, j) for i, j in zip(t_channel_pearson, s_channel_pearson))
        channel_relation_loss = sum(self.loss(i, j) for i, j in zip(t_channel_relation, s_channel_relation))

        sample_pearson_loss = sum(self.loss(i, j) for i, j in zip(t_sample_pearson, s_sample_pearson))
        sample_relation_loss = sum(self.loss(i, j) for i, j in zip(t_sample_relation, s_sample_relation))

        loss = [spatial_pearson_loss, spatial_relation_loss, channel_pearson_loss, channel_relation_loss,
                sample_pearson_loss, sample_relation_loss]

        factor = F.softmax(torch.Tensor(loss), dim=-1)
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))

        return loss_t


if __name__ == '__main__':
    # pass
    x = torch.randn(64, 3, 32, 32)

    b, _, _, _ = x.shape

    s_net = resnet8x4(num_classes=100)

    t_net = resnet32x4(num_classes=100)

    s_feats, s_logit = s_net(x, is_feat=True, preact=False)
    t_feats, t_logit = t_net(x, is_feat=True, preact=False)

    with torch.no_grad():
        s_sample_pearson = SKD.stage_channel_pearson(s_feats[:-1])
