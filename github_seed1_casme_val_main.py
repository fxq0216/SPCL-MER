import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import resnet
import pandas as pd
import os
import github_seed1_casme_supcon_dataset
from torch.utils.tensorboard import SummaryWriter
from random import sample
import pandas as pd
from github_seed1_casme_main_linear import *
import Metrics as metrics
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def linear_m_eval(opt, sub_id):
    # build data loader
    train_loader, val_loader = github_seed1_casme_supcon_dataset.getdata(sub_id, mode='linear')
    # build model and criterion
    model, classifier, classifier_val, criterion, criterion_center = set_model(opt, sub_id)

    # eval
    classifier_val.load_state_dict(torch.load(f"./casme_seed1/{sub_id}_best_linear.pth", map_location='cpu'))
    fea_center, target = compute_center(train_loader, model, criterion_center, opt)
    fea_center = fea_center.cuda()
    target = target.cuda()
    loss, val_acc, uf1, output, labels = validate(val_loader, model, classifier_val, criterion, opt, fea_center, target, sub_id)
    print('Subject {}, val_loss{:.2f}, val_acc {:.2f}'.format(sub_id, loss, val_acc))
    print('\tSubject {} has the ACC:{:.4f},UF1:{:.2f}\n'.format(sub_id, val_acc, uf1))
    print('---------------------------\n')
    return val_acc, uf1, output, labels


def main():
    opt = parse_option()
    #LOSO
    df = pd.read_table("dataset/casme2/csame_train.log", sep=',', header=None)
    sub_id = df.iloc[:, 2].values #文件中的subid列
    subjects = list(set(sub_id)) #sub_id的集合
    sampleNum = len(subjects) #sub的个数
    mean_acc = 0
    all_output = []
    all_labels = []

    with open("log_test_three_casme_12.16.txt", "w") as f2:
        for id in subjects:
            # 训练
            # if id!=10 and id!=18 and id!=21:#four
            if id!=10 and id!=18:#three
                if __name__ == "__main__":
                    #project head
                    acc,_,output,labels= linear_m_eval(opt, id)#分类器训练
                    mean_acc += acc
                    if acc !=0:
                        all_output.extend(output)
                        all_labels.extend(labels)

                    common_elements = sum([1 for i, j in zip(labels, output) if i == j])
                    # print(common_elements)
                    print(f'total_num:{len(labels)}, correct_num:{common_elements}')
                    f2.write(f'subid:{id}\n')
                    f2.write(f'total_num:{len(labels)}, correct_num:{common_elements}\n')
                    f2.write('LOSO_id: %03d | Acc: %.3f%%'
                            % (id, acc))
                    f2.write('\n')
                    f2.flush()


        print("Finished--------------------------------------------------")
        ##评价指标
        pre=torch.tensor(all_output)
        lab=torch.tensor(all_labels)
        eval_acc = metrics.accuracy()
        eval_f1 = metrics.f1score()
        acc_w, acc_uw = eval_acc.eval(pre,lab)
        _, f1_uw = eval_f1.eval(pre,lab)
        print('\nThe dataset has the ACC :{:.4f}'.format(acc_w))
        print('\nThe dataset has the UAR and UF1:{:.4f} and {:.4f}'.format(acc_uw, f1_uw))

if __name__ == '__main__':
    main()


