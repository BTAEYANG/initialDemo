import torch
import torch.nn as nn
import torch.nn.functional as F


class GKD(nn.Module):
    """Guided Knowledge Distillation Loss"""

    def __init__(self, T):
        super(GKD, self).__init__()
        self.t = T

    def forward(self, y_s, y_t, label):
        s = F.log_softmax(y_s / self.t, dim=1)
        t = F.softmax(y_t / self.t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        mask = torch.eq(label, t_argmax).float()
        count = (mask[mask == 1]).size(0)
        mask = mask.unsqueeze(-1)
        error_index = [i for i, x in enumerate(mask) if x == 0]
        correct_s = s.mul(mask)
        correct_t = t.mul(mask)
        # correct_t[correct_t == 0.0] = 1.0

        loss = F.kl_div(correct_s, correct_t, reduction='sum') * (self.t ** 2) / count
        return loss, count, error_index


if __name__ == '__main__':
    a = torch.randn(64, 32, 32)
    b = torch.zeros(64, 32, 32)


