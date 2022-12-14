"""
the general training framework
"""

from __future__ import print_function

import csv
import os
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataset.cifar import getDataLoader
from distillation_zoo.KD import DistillKL
from models import model_dict
from paper.DWD import DWD, DWD_Loss
from paper.FPD import FPD, FPD_Loss
from paper.GKD import GKD
from paper.GSKD import GSKD, GSKD_Loss
from paper_skd.SKD import SKD, SKD_Loss
from util.embedding_util import LinearEmbed, MLPEmbed
from util.tool import adjust_learning_rate, get_teacher_name, load_teacher, adjust_beta_rate
from util.train_loops import train_distill
from util.val_loops import validate


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['CIFAR100', 'CIFAR10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4',
                                 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'FPD', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst', 'SKD', 'DWD', 'GSKD'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--loss_type', type=str, default='MSE', help='choose loss-type function')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=10, help='weight balance for other losses')

    parser.add_argument('--kd_type', type=str, default='KD', help='choose KD-loss type')

    parser.add_argument('--beta_increase_rate', type=float, default=1,
                        help='increase rate for beta loss -b??? default 1 beta not change')
    parser.add_argument('--beta_decay_rate', type=float, default=1,
                        help='decay rate for beta loss -b??? default 1 beta not change')
    parser.add_argument('--beta_rate_epochs', type=str, default='90,120,150,180,210',
                        help='where to change beta, can be a list')
    parser.add_argument('--new_beta', type=float, default=None, help='record new weight balance for other losses')
    parser.add_argument('--reverse', default='False', action='store_true', help='reverse loss factor')
    parser.add_argument('--cuda_id', type=int, default=3, help='cuda run id')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    beta_iterations = opt.beta_rate_epochs.split(',')
    opt.beta_rate_epochs = list([])
    for it in beta_iterations:
        opt.beta_rate_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'Student:{}_Teacher:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset,
                                                                            opt.distill,
                                                                            opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    if opt.distill == 'SKD':
        torch.cuda.set_device(opt.cuda_id)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    train_loader, val_loader, n_cls = getDataLoader(opt)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)

    model_t.eval()
    model_s.eval()

    feat_t, _ = model_t(data, is_feat=True, preact=False)
    feat_s, _ = model_s(data, is_feat=True, preact=False)

    # init module list to add student model and teacher model or other model need by some kd methods
    module_list = nn.ModuleList([])
    module_list.append(model_s)  # add student model

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()

    if opt.kd_type == 'KD':
        criterion_kl = DistillKL(opt.kd_T)
    elif opt.kd_type == 'GKD':
        criterion_kl = GKD(opt.kd_T)
    else:
        criterion_kl = GKD(opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'FPD':
        fpd = FPD(feat_t, feat_s)
        module_list.append(fpd)
        trainable_list.append(fpd)
        criterion_kd = FPD_Loss()
    elif opt.distill == 'DWD':
        dwd = DWD(feat_t[:-1], feat_s[:-1])
        module_list.append(dwd)
        trainable_list.append(dwd)
        criterion_kd = DWD_Loss(opt.loss_type)
    elif opt.distill == 'SKD':

        embed_s = nn.ModuleList([])
        embed_t = nn.ModuleList([])

        feat_t = feat_t[:-1]
        feat_s = feat_s[:-1]

        dim_in_t = []
        dim_in_s = []

        for i in range(len(feat_t) - 1):
            b_H, t_H = feat_t[i].shape[2], feat_t[i + 1].shape[2]
            if b_H > t_H:
                dim_in_t.append(int(t_H * t_H))
            else:
                dim_in_t.append(int(b_H * b_H))

        for k in range(len(feat_s) - 1):
            s_b_H, s_t_H = feat_s[k].shape[2], feat_s[k + 1].shape[2]
            if s_b_H > s_t_H:
                dim_in_s.append(int(s_t_H * s_t_H))
            else:
                dim_in_s.append(int(s_b_H * s_b_H))

        for t, s in zip(dim_in_t, dim_in_s):
            if feat_s[-1].shape[1] == feat_t[-1].shape[1]:
                embed_s.append(MLPEmbed(dim_in=t, dim_out=feat_t[-1].shape[1]))
                embed_t.append(MLPEmbed(dim_in=t, dim_out=feat_t[-1].shape[1]))
            else:
                embed_s.append(MLPEmbed(dim_in=s, dim_out=feat_t[-1].shape[1]))
                embed_t.append(MLPEmbed(dim_in=t, dim_out=feat_t[-1].shape[1]))

        skd = SKD()

        module_list.append(skd)
        module_list.append(embed_s)
        module_list.append(embed_t)

        trainable_list.append(skd)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)

        criterion_kd = SKD_Loss(opt.loss_type)

    elif opt.distill == 'GSKD':
        feat_s = feat_s[:-1]
        embed_s = nn.ModuleList([])
        for s in feat_s:
            h = s.shape[2]
            embed_s.append(MLPEmbed(dim_in=h * h, dim_out=feat_t[-1].shape[1]))

        gskd = GSKD()

        module_list.append(gskd)
        module_list.append(embed_s)

        trainable_list.append(gskd)
        trainable_list.append(embed_s)

        criterion_kd = GSKD_Loss(opt.loss_type)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss using CrossEntropyLoss
    criterion_list.append(criterion_kl)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss, just like the kd method using DistillKL

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('Validate Teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        new_lr = adjust_learning_rate(epoch, opt, optimizer)

        # beta from initial value increase to final value with epoch (0.01 - 0.1 - 1 -10)
        new_beta = adjust_beta_rate(epoch, opt)
        print(f"==> Training... Current lr: {new_lr}; Current -b: {new_beta}")

        time1 = time.time()
        train_acc, train_loss = train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt,
                                              new_beta)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            csv_name = os.path.join(opt.save_folder,
                                    f'{opt.model_s}_{opt.model_t}_{opt.dataset}_{opt.distill}_r:{opt.gamma}_a:{opt.alpha}_b:{opt.beta}_T:{opt.kd_T}_{opt.trial}_best.csv')
            print('saving the best model and csv!')
            torch.save(state, save_file)
            with open(csv_name, mode='a', newline='', encoding='utf8') as csv_file:
                csv_writer = csv.writer(csv_file)
                # columns_name
                csv_writer.writerow(
                    ["model_s", "model_t", "dataset", "distill", "loss_type", "trial", "epoch", "new_bate", "best_acc"])

                csv_writer.writerow([opt.model_s, opt.model_t, opt.dataset, opt.loss_type,
                                     opt.distill, opt.trial, epoch, new_beta, best_acc])
                csv_file.close()

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
