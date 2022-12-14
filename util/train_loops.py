import sys
import time
import torch

from util.tool import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, new_beta):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_kl = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):

        data_time.update(time.time() - end)
        input, target = data

        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        preact = False
        # student model output : student feature map and student logit value
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            # teacher model output : teacher feature map and teacher logit value
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            # ??????????????????tensor?????????????????????????????????????????????????????????????????????????????????,??????????????????requires_grad???false??????????????????tensor??????????????????????????????????????????grad
            feat_t = [f.detach() for f in feat_t]

        # cls -r weight for classification
        loss_cls = criterion_cls(logit_s, target)
        # kl  -a weight for KD distillation  GKD get loss, correct count, error_index
        if opt.kd_type == 'GKD':
            loss_kl, error_index = criterion_kl(logit_s, logit_t, target)
        else:
            loss_kl = criterion_kl(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'FPD':
            f_t, f_s, g_t, g_s, se_g_t, se_g_s = module_list[1](feat_t, feat_s, error_index)
            loss_kd = criterion_kd(f_t, f_s, g_t, g_s, se_g_t, se_g_s)
        elif opt.distill == 'DWD':
            f_t, f_s, f_t_res, f_s_res = module_list[1](feat_t[:-1], feat_s[:-1])
            loss_kd = criterion_kd(f_t, f_s, f_t_res, f_s_res, opt)
        elif opt.distill == 'SKD':
            embed_s = module_list[2]
            embed_t = module_list[3]
            t_tensor, s_tensor, t_fc_tensor, s_fc_tensor = module_list[1](feat_t[:-1], feat_s[:-1], embed_s, embed_t, model_t, opt)
            loss_kd = criterion_kd(t_tensor, s_tensor, t_fc_tensor, s_fc_tensor, opt)
        elif opt.distill == 'GSKD':
            embed_s = module_list[2]
            tensor_l = module_list[1](logit_t, target, feat_s[:-1], model_t, opt, embed_s)
            loss_kd = criterion_kd(tensor_l, input.size(0), opt)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_kl + new_beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg
