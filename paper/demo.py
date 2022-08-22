import torch
import torch.nn.functional as F
from torch import nn


def avg(x, y):
    loss = 0.
    loss_2 = 0.
    avg_pool = nn.AdaptiveAvgPool2d(1)
    for s, t in zip(x, y):
        print(s.shape, t.shape)
        s = avg_pool(s)
        t = avg_pool(t)
        print(s.shape, t.shape)
        loss += torch.mean(torch.pow(s - t, 2))
        loss_2 += F.mse_loss(s, t, reduction='mean')
    return loss, loss_2


def _spatial_mean_loss(x, y):
    loss = 0.
    for s, t in zip(x, y):
        s = s.mean(dim=1, keepdim=False)
        t = t.mean(dim=1, keepdim=False)
        loss += F.mse_loss(s, t, reduction='mean')
    return loss


if __name__ == "__main__":
    pass