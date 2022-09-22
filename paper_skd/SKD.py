import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    def forward(self, f_t, f_s):
        s_pearson = self.cov_pearson(f_s)

        with torch.no_grad():
            t_pearson = self.cov_pearson(f_t)

        return t_pearson, s_pearson

    @staticmethod
    def cov_pearson(f):

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


class SKD_Loss(nn.Module):
    """ Stage Relation Distilling Loss"""

    def __init__(self):
        super(SKD_Loss, self).__init__()
        # self.huber = nn.HuberLoss(reduction='mean', delta=1.0)
        self.smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, t_pearson, s_pearson):

        loss_stage = [self.smoothL1(i, j) for i, j in zip(t_pearson, s_pearson)]
        factor = F.softmax(torch.Tensor(loss_stage), dim=-1)
        loss = sum(factor[index] * loss_stage[index] for index, value in enumerate(loss_stage))

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

        SKD.cov_pearson(s_feats[:-1])

