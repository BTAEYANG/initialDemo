import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.cifar import getDataLoader
import argparse
from paper.GKD import GKD
from train_student import parse_option


class GSKD(nn.Module):
    """Guided Similarity Knowledge Distillation"""

    def __init__(self):
        super(GSKD, self).__init__()

    @staticmethod
    def _get_error_index(y_t, label):
        t = F.softmax(y_t, dim=1)
        t_argmax = torch.argmax(t, dim=1)
        mask = torch.eq(label, t_argmax).float()
        mask = mask.unsqueeze(-1)
        error_index = [i for i, x in enumerate(mask) if x == 0]
        return error_index

    def forward(self, y_s, y_t, label):
        error_index = self.get_error_index(y_t, label)
        if len(error_index):
            for s, t in zip(y_s, y_t):
                for j in error_index:
                    s[j].data *= 0
                    t[j].data *= 0

        y_s = y_s.mean(dim=1, keepdim=False)
        y_t = y_t.mean(dim=1, keepdim=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['CIFAR100', 'CIFAR10'], help='dataset')
    opt = parser.parse_args()

    train_loader, val_loader, n_cls = getDataLoader(opt)

    for idx, data in enumerate(train_loader):
        input, target = data
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        print(input.shape)
        print(target.shape)
