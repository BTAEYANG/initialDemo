import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    def forward(self, f_t, f_s):
        s_stage_pearson = self.stage_pearson(f_s)
        s_sample_pearson = self.sample_pearson(f_s)

        with torch.no_grad():
            t_stage_pearson = self.stage_pearson(f_t)
            t_sample_pearson = self.sample_pearson(f_t)

        return t_stage_pearson, s_stage_pearson, t_sample_pearson, s_sample_pearson

    @staticmethod
    def stage_pearson(f):

        for i in range(len(f)):
            f[i] = f[i].mean(dim=1, keepdim=False).view(f[i].shape[0], -1).unsqueeze(1)

        matrix_list = []
        for i in range(len(f) - 1):
            matrix_list.append((torch.bmm(f[i].transpose(1, 2), f[i + 1])).mean(dim=0, keepdim=False))

        pearson_list = []
        for m in matrix_list:
            if m.shape[0] == m.shape[1]:
                pearson_list.append(torch.corrcoef(m))
            else:
                pearson_list.append(torch.corrcoef(m))
                pearson_list.append(torch.corrcoef(m.t()))

        return pearson_list

    @staticmethod
    def sample_pearson(f):
        avg_pool = nn.AdaptiveAvgPool2d(1)
        for i in range(len(f)):
            f[i] = avg_pool(f[i].mean(dim=1, keepdim=False).unsqueeze(0)).squeeze(2)

        sample_matrix_list = []
        for i in range(len(f) - 1):
            sample_matrix_list.append((torch.bmm(f[i], f[i + 1].transpose(1, 2)).squeeze(0)))

        sample_pearson_list = []
        for m in sample_matrix_list:
            sample_pearson_list.append(torch.corrcoef(m))

        return sample_pearson_list


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

    def forward(self, t_stage_pearson, s_stage_pearson, t_sample_pearson, s_sample_pearson):

        stage_loss = sum(self.loss(i, j) for i, j in zip(t_stage_pearson, s_stage_pearson))
        sample_stage_loss = sum(self.loss(i, j) for i, j in zip(t_sample_pearson, s_sample_pearson))

        total_stage_loss = stage_loss + sample_stage_loss

        loss = (stage_loss * (sample_stage_loss / total_stage_loss)) + (
                    sample_stage_loss * (stage_loss / total_stage_loss))

        return loss


if __name__ == '__main__':
    pass
    x = torch.randn(10, 3, 32, 32)

    b, _, _, _ = x.shape

    s_net = resnet8x4(num_classes=100)

    t_net = resnet32x4(num_classes=100)

    s_feats, s_logit = s_net(x, is_feat=True, preact=True)
    t_feats, t_logit = t_net(x, is_feat=True, preact=True)

    with torch.no_grad():
        SKD.sample_pearson(s_feats[:-1])
