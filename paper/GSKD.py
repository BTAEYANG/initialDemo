import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.cifar import getDataLoader
import argparse

from models import mobile_half, resnet32x4
from paper.GKD import GKD
from train_student import parse_option
from util.embedding_util import MLPEmbed


class GSKD(nn.Module):
    """Guided Similarity Knowledge Distillation"""

    def __init__(self, y_s, y_t):
        super(GSKD, self).__init__()
        self.embed_s = nn.ModuleList([])
        for s in y_s:
            h = s.shape[2]
            self.embed_s.append(MLPEmbed(dim_in=h * h, dim_out=y_t.shape[1]))

    @staticmethod
    def _get_error_index(y_t, label):
        t = F.softmax(y_t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        error = torch.eq(label, t_argmax).float()
        error = error.unsqueeze(-1)
        error_index = [i for i, x in enumerate(error) if x == 0]
        return error_index

    def forward(self, y_t, label, f_s, model_t, opt):
        error_index = self.get_error_index(y_t, label)
        if len(error_index):
            for s in f_s:
                for j in error_index:
                    s[j].data *= 0
        tensor_l = []
        for i, f in enumerate(f_s):
            b, c, h, w = f.shape
            f = f.mean(dim=1, keepdim=False).view(b, h * w)
            if opt.model_t.__contains__('vgg'):
                f = model_t.classifier(self.embed_s[i](f))
            elif opt.model_t.__contains__('ResNet'):
                f = model_t.linear(self.embed_s[i](f))
            else:
                f = model_t.fc(self.embed_s[i](f))
            tensor_l.append(f @ y_t.t())  # 64 * 64

        return tensor_l


class GSKD_Loss(nn.Module):
    """ Guided Similarity Knowledge Distilling Loss"""

    def __init__(self, loss_type, batch_size):
        super(GSKD_Loss, self).__init__()

        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss(reduction='mean', delta=1.0)
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

        self.s_matrix = torch.eye(batch_size)

    def forward(self, tensor_l, opt):
        loss = [self.loss(i, self.s_matrix) for i in tensor_l]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        if opt.reverse:
            loss.reverse()
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))
        return loss_t


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser('argument for training')
    # parser.add_argument('--model_t', type=str, default='', help='')
    # opt = parser.parse_args()
    #
    # x = torch.randn(64, 3, 32, 32)
    #
    # s_net = mobile_half(num_classes=100)
    # t_net = resnet32x4(num_classes=100)
    #
    # f_s, s_logit = s_net(x, is_feat=True, preact=False)
    # f_t, t_logit = t_net(x, is_feat=True, preact=False)
    #
    # gskd = GSKD(f_s[:-1], f_t[-1])
    #
    # gskd(t_logit, f_s[:-1], t_net, opt)
