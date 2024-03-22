"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss_pair(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss_pair, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)#mask是正类
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # print(mask * log_prob)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class centercompute(nn.Module):
    def __init__(self):
        super(centercompute, self).__init__()
        self.happy=[]
        self.sur=[]
        self.repssion=[]
        self.other=[]
# 
    def forward(self, features, labels, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        self.happy=[]
        self.sur=[]
        self.repssion=[]
        self.other=[]

        for f, l in zip(range(features.size(0)), labels):
            if l==0:
                self.happy.append(features[f,:])
            if l==1:
                self.repssion.append(features[f,:])
            if l==2:
                self.sur.append(features[f,:])
            if l==3:
                self.other.append(features[f,:])

        if len(self.happy)>0:
            h=torch.stack(self.happy, dim=0)
            hc = (sum(h))/(len(self.happy))

        if len(self.repssion)>0:
            r=torch.stack(self.repssion, dim=0)
            rc = (sum(r))/(len(self.repssion))

        if len(self.sur)>0:
            s=torch.stack(self.sur, dim=0)
            sc = (sum(s))/(len(self.sur))

        if len(self.other)>0:
            o=torch.stack(self.other, dim=0)
            oc = (sum(o))/(len(self.other))

        fea_center = torch.stack((hc, rc, sc, oc), dim=0)
        fea_center = F.normalize(fea_center, dim=1)
        target = torch.tensor([0,1,2,3])
        return fea_center, target



class proLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all'):
        super(proLoss, self).__init__()
        self.contrast_mode = contrast_mode

    def forward(self, features, fea_center, target, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]*2
        labels = labels.contiguous().view(-1, 1)
        target = target.unsqueeze(-1)
        mask = torch.eq(labels, target.T).float().to(device)#mask是正类

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#恢复成两个维度
        anchor_feature = fea_center
        anchor_count = 1
        anchor_dot_contrast = torch.matmul(contrast_feature, anchor_feature.T)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)#应该是对角线的值全是1
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def cosine(x, y):
    '''
    Compute cosine similiarity between two tensors
    '''
    # x: N x D
    # y: M x D
    cos = torch.matmul(x, y.T)#batchsize*5clas
    return cos

class centercompute_val(nn.Module):
    def __init__(self):
        super(centercompute_val, self).__init__()
        self.happy=[]
        self.sur=[]
        self.repssion=[]
#
    def forward(self, features, labels, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        self.happy=[]
        self.sur=[]
        self.repssion=[]

        for f, l in zip(range(features.size(0)), labels):
            if l==0:
                self.happy.append(features[f,:])
            if l==1:
                self.repssion.append(features[f,:])
            if l==2:
                self.sur.append(features[f,:])

        if len(self.happy)>0:
            h=torch.stack(self.happy, dim=0)
            hc = (sum(h))/(len(self.happy))

        if len(self.repssion)>0:
            r=torch.stack(self.repssion, dim=0)
            rc = (sum(r))/(len(self.repssion))

        if len(self.sur)>0:
            s=torch.stack(self.sur, dim=0)
            sc = (sum(s))/(len(self.sur))

        fea_center = torch.stack((hc, rc, sc), dim=0)
        fea_center = F.normalize(fea_center, dim=1)
        target = torch.tensor([0,1,2])
        return fea_center, target

def proloss(output, fea_center):
    dists = cosine(output, fea_center)  # 计算距离
    # 计算cos时，cos在-1-》1之间，cos越大越相似
    log_p_y = F.log_softmax(dists, dim=1)
    return log_p_y


