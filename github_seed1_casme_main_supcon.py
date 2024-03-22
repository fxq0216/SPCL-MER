from __future__ import print_function
import os
import sys
import argparse
import time
import math
import github_seed1_casme_supcon_dataset
from torch.utils.tensorboard import SummaryWriter
import torch
torch.cuda.current_device()
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np
import random
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate,accuracy
from util import set_optimizer, save_model
from context_cluster_attv1 import SupConcluster
import torch.nn.functional as F
from loss_center_three_sup import *
import interpretdl as it

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    opt = parser.parse_args()

    return opt

def set_model(opt):
    model = Cocatt()
    criterion_pro = proLoss(temperature=opt.temp)
    criterion_center = centercompute(temperature=opt.temp)
    criterion_supcon = SupConLoss_pair(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        criterion_pro = criterion_pro.cuda()
        criterion_center = criterion_center.cuda()
        criterion_supcon = criterion_supcon.cuda()
        cudnn.benchmark = True

    return model, criterion_pro, criterion_center, criterion_supcon



def train(train_loader, model, criterion_pro, criterion_pair, optimizer, epoch, opt, fea_center, target):
    """one epoch training"""
    model.train()
    losses = AverageMeter()
    # acc=AverageMeter()

    for idx, data in enumerate(train_loader):
        images, labels, index=data
        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        labels_l = labels.repeat(2, 1)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)#(20,128)(20,128)  将tensor分成块结构,[,]为切分后的大小，dim为切分的维度
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)#(20,2,128)

        loss_pro = criterion_pro(features, fea_center, target, labels_l)
        loss_supcon = criterion_pair(features, labels)
        loss = loss_pro+loss_supcon

        # update metric
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: epoch：{0},batch_num：[{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   .format(epoch, idx + 1, len(train_loader),loss=losses))
            sys.stdout.flush()

    return losses.avg

def compute_center(train_loader, model, criterion_center):
    """obtain protype center"""
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            images, labels, index=data
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            labels = labels.repeat(2)
            
            features = model(images)
            if idx==0:
                res_fea = features
                res_label = labels
            else:
                res_fea = torch.cat((res_fea, features), dim=0)
                res_label = torch.cat((res_label, labels), dim=0)
    fea_center, target = criterion_center(res_fea, res_label)
    return fea_center, target


def supcon_m(sub_id):
    # 获取当前的seed
    current_seed = torch.initial_seed()
    # 打印当前的seed
    print('-----------------------------------------------')
    print("Current seed:", current_seed)
    opt = parse_option()
    writer = SummaryWriter("con_log")
    # build data loader
    train_loader= github_seed1_casme_supcon_dataset.getdata(sub_id, mode='supcon')  # 读取并加载数据集
    # build model and criterion
    model, criterion_pro, criterion_center, criterion_supcon = set_model(opt)#
    optimizer = set_optimizer(opt, model) #优化器
    best_loss = 100

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        time1 = time.time()
        fea_center, target = compute_center(train_loader, model, criterion_center)
        fea_center = fea_center.cuda()
        target = target.cuda()
        loss = train(train_loader, model, criterion_pro, criterion_supcon, optimizer, epoch, opt, fea_center, target)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        #save_premodel
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, f'ckpt_epoch_{epoch}_{sub_id}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    torch.save(model.state_dict(), f'casme_seed1/{sub_id}_last.pth')
    print("Save model")

    writer.close()
    return loss

if __name__ == '__main__':
    init_seed = 1
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    np.random.seed(init_seed)
    random.seed(init_seed)
    
    for i in range(26):
        supcon_m(i+1)
 