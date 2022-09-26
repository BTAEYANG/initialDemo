import torch
from torch import nn
import torch.nn.functional as F

from models import resnet32x4, resnet8x4


class SKD(nn.Module):

    def __init__(self):
        super(SKD, self).__init__()

    def forward(self, f_t, f_s):

        B_relation_list_s, C_relation_list_s, H_relation_list_s, W_relation_list_s = self.stage_sample_relation(f_s)

        with torch.no_grad():
            B_relation_list_t, C_relation_list_t, H_relation_list_t, W_relation_list_t = self.stage_sample_relation(f_t)

        relation_list_s = [B_relation_list_s, C_relation_list_s, H_relation_list_s, W_relation_list_s]
        relation_list_t = [B_relation_list_t, C_relation_list_t, H_relation_list_t, W_relation_list_t]

        return relation_list_s, relation_list_t

    @staticmethod
    def stage_sample_relation(f):

        temp_B = []
        temp_C = []
        temp_H = []
        temp_W = []
        for i in range(len(f)):
            temp_B.append(f[i].mean(dim=1, keepdim=False).mean(dim=0, keepdim=False).view(-1).unsqueeze(0))
            temp_C.append(f[i].mean(dim=1, keepdim=False).view(f[i].shape[0], -1))
            temp_H.append(f[i].mean(dim=2, keepdim=False).mean(dim=1, keepdim=False))
            temp_W.append(f[i].mean(dim=3, keepdim=False).mean(dim=1, keepdim=False))

        B_relation_list = []
        C_relation_list = []
        H_relation_list = []
        W_relation_list = []
        for j in range(len(temp_C) - 1):
            B_relation_list.append(temp_B[j].transpose(0, 1) @ temp_B[j + 1])
            C_relation_list.append(temp_C[j].transpose(0, 1) @ temp_C[j + 1])
            H_relation_list.append(temp_H[j].transpose(0, 1) @ temp_H[j + 1])
            W_relation_list.append(temp_W[j].transpose(0, 1) @ temp_H[j + 1])

        return B_relation_list, C_relation_list, H_relation_list, W_relation_list


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

    def forward(self, relation_list_s, relation_list_t):

        b_loss = sum(self.loss(i, j) for i, j in zip(relation_list_s[0], relation_list_t[0]))
        c_loss = sum(self.loss(i, j) for i, j in zip(relation_list_s[1], relation_list_t[1]))
        w_loss = sum(self.loss(i, j) for i, j in zip(relation_list_s[2], relation_list_t[2]))
        h_loss = sum(self.loss(i, j) for i, j in zip(relation_list_s[3], relation_list_t[3]))

        relation_loss = [b_loss, c_loss, w_loss, h_loss]
        factor = F.softmax(torch.Tensor(relation_loss), dim=-1)

        loss = sorted(relation_loss)
        factor = sorted(factor.tolist(), reverse=True)

        relation_loss_t = sum(factor[index] * relation_loss[index] for index, value in enumerate(relation_loss))

        return relation_loss_t


if __name__ == '__main__':
    pass
    x = torch.randn(64, 3, 32, 32)

    b, _, _, _ = x.shape

    s_net = resnet8x4(num_classes=100)

    t_net = resnet32x4(num_classes=100)

    s_feats, s_logit = s_net(x, is_feat=True, preact=False, feat_preact=True)
    t_feats, t_logit = t_net(x, is_feat=True, preact=True, feat_preact=True)

    f_s = s_feats[:-1]

    with torch.no_grad():
        SKD.stage_sample_relation(f_s)
        SKD.stage_sample_relation(f_s)
