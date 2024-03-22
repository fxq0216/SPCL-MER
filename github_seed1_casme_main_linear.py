
from __future__ import print_function

import sys
import argparse
import time
import math
import os
import torch
import torch.backends.cudnn as cudnn
import github_seed1_casme_supcon_dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, f1_score
# from resnet import SupConResNet, LinearClassifier
from context_cluster_attv1 import SupConcluster, LinearClassifier
import Metrics as metrics
import torch.nn.functional as F
from loss_center_three_sup import *
from util import save_model
from collections import Counter

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=2,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=30,  # 20
                        help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--ckpt', type=str, default=f'./casme_seed1/{id}_last.pth')
    opt = parser.parse_args()
    return opt



def set_model(opt, id):

    model = Cocatt()
    criterion = torch.nn.CrossEntropyLoss()
    criterion_center = centercompute_val()

    classifier = LinearClassifier(name=opt.model, num_classes=128)
    classifier_val = LinearClassifier(name=opt.model, num_classes=128)
    opt.ckpt = f'./casme_seed1/{id}_last.pth'

    print(opt.ckpt)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 7:  # 1
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        classifier_val = classifier_val.cuda()
        criterion = criterion.cuda()
        criterion_center = criterion_center.cuda()
        cudnn.benchmark = True

    model.load_state_dict(state_dict)
    return model, classifier, classifier_val, criterion, criterion_center


def compute_center(train_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(train_loader):
            images, labels, index = data
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            features = model(images)
            if idx == 0:
                res_fea = features
                res_label = labels#labels.repeat(2)
            else:
                res_fea = torch.cat((res_fea, features), dim=0)
                res_label = torch.cat((res_label, labels), dim=0)#labels.repeat(2)
    fea_center, target = criterion(res_fea, res_label)
    return fea_center, target


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, writer, fea_center, target, id):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, data in enumerate(train_loader):
        images, labels, index = data
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        output = F.normalize(output, dim=1)
        output = proloss(output, fea_center)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 2))
        top1.update(acc1[0], bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


    torch.save(classifier.state_dict(), f'casme_seed1/{id}_train.pth')
    writer.add_scalar("train_loss", losses.avg, epoch)
    writer.add_scalar("train_acc", top1.avg, epoch)

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, fea_center, target, id):
    """validation"""
    model.eval()
    classifier.eval()

    ##评价指标1
    losses = AverageMeter()
    top1 = AverageMeter()
    UF1 = AverageMeter()
    all_output = []
    all_labels = []

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            images, labels, index = data
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output = classifier(model.encoder(images))

            #某一sub中只有一个样本
            if len(output.shape)==1:
                output = output.unsqueeze(dim=0)

            output = F.normalize(output, dim=1)
            output = proloss(output, fea_center)
            loss = criterion(output, labels)
            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 2))
            top1.update(acc1[0], bsz)
            _, uf1 = f1_score(output, labels)
            UF1.update(uf1 * 100, bsz)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            output_v = torch.squeeze(pred, dim=0)
            all_output.extend(output_v.tolist())
            all_labels.extend(labels.tolist())
    return losses.avg, top1.avg, UF1.avg, all_output, all_labels




def linear_m(sub_id):
    opt = parse_option()
    writer = SummaryWriter('con_log')
    # build data loader
    train_loader, val_loader = github_seed1_casme_supcon_dataset.getdata(sub_id, mode='linear')
    # build model and criterion
    model, classifier, classifier_val, criterion, criterion_center = set_model(opt, sub_id)
    # build optimizer
    optimizer= torch.optim.Adam(classifier.parameters(),lr=0.0001,betas=(0.9,0.99))

    best_acc = 0
    best_uf1 = 0
    best_output = np.zeros((opt.n_cls, opt.n_cls), int).tolist()  # 创建n维0矩阵，转list
    best_labels = best_output

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        fea_center, target = compute_center(train_loader, model, criterion_center)
        fea_center = fea_center.cuda()
        target = target.cuda()
        train_loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt, writer, fea_center, target, sub_id)
        print('Subject {}, epoch {}, train_loss{:.2f}, train_acc {:.2f}'.format(sub_id, epoch, train_loss, train_acc))

        # eval
        classifier_val.load_state_dict(torch.load(f"./casme_seed1/{sub_id}_train.pth", map_location='cpu'))
        fea_center, target = compute_center(train_loader, model, criterion_center)
        fea_center = fea_center.cuda()
        target = target.cuda()
        loss, val_acc, uf1, output, labels = validate(val_loader, model, classifier_val, criterion, opt, fea_center, target, sub_id)
        print('Subject {}, epoch {}, val_loss{:.2f}, val_acc {:.2f}'.format(sub_id, epoch, loss, val_acc))


        if val_acc > best_acc:
            best_acc = val_acc
            best_output = output
            best_labels = labels
            best_uf1 = uf1
            torch.save(classifier.state_dict(), f'casme_seed1/{sub_id}_best_linear.pth')
            print("Save model")
    print('\tSubject {} has the ACC:{:.4f},UF1:{:.2f}\n'.format(sub_id, best_acc, best_uf1))
    print('---------------------------\n')
    return best_acc, best_uf1, best_output, best_labels


if __name__ == '__main__':
    linear_m(8)

