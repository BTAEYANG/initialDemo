import torch
import torch.nn as nn
import torch.nn.functional as F

from util.embedding_util import MLPEmbed


class GSKD(nn.Module):
    """Guided Similarity Knowledge Distillation"""

    def __init__(self):
        super(GSKD, self).__init__()

    @staticmethod
    def _get_error_index(y_t, label):
        t = F.softmax(y_t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        error = torch.eq(label, t_argmax).float()
        error = error.unsqueeze(-1)
        error_index = [i for i, x in enumerate(error) if x == 0]
        return error_index

    def forward(self, y_t, label, f_s, model_t, opt, embed_s):
        error_index = self._get_error_index(y_t, label)
        if len(error_index):
            for s in f_s:
                for j in error_index:
                    s[j].data *= 0
        tensor_l = []
        for i, f in enumerate(f_s):
            b, c, h, w = f.shape
            f = f.mean(dim=1, keepdim=False).view(b, h * w)
            if opt.model_t.__contains__('vgg'):
                f = model_t.classifier(embed_s[i](f))
            elif opt.model_t.__contains__('ResNet'):
                f = model_t.linear(embed_s[i](f))
            else:
                f = model_t.fc(embed_s[i](f))
            tensor_l.append(torch.softmax((f @ y_t.t()), dim=0))  # 64 * 64

        return tensor_l


class GSKD_Loss(nn.Module):
    """ Guided Similarity Knowledge Distilling Loss"""

    def __init__(self, loss_type):
        super(GSKD_Loss, self).__init__()
        if loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        elif loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss(reduction='mean', delta=1.0)
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

    def forward(self, tensor_l, opt):
        s_matrix = torch.eye(opt.batch_size)
        loss = [self.loss(i, s_matrix) for i in tensor_l]
        factor = F.softmax(torch.Tensor(loss), dim=-1)
        if opt.reverse:
            loss.reverse()
        loss_t = sum(factor[index] * loss[index] for index, value in enumerate(loss))
        return loss_t


if __name__ == '__main__':
    pass
