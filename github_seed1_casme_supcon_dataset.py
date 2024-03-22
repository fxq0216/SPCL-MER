import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import torch
import os
import argparse
from util import TwoCropTransform, AverageMeter
from PIL import Image
from  torchvision import utils as vutils
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')#256,调小batch——size防止过拟合
    parser.add_argument('--workers', type=int, default=4)#数据集较小时，不需要多个num_work
    parser.add_argument('--datasets_csv',type=str,default='dataset/casme2/csame_train.log',
                        help='casme2/c3_train.csv,casme2/c5_train.csv,'
                             'samm/samm_c3_train.csv,samm/samm_c3_train.csv'
                             'smic/smic_c3_train.csv')
    parser.add_argument('--mode', type=str, default='supcon')
    return parser.parse_args()


class casme2DataSet(data.Dataset):
    def __init__(self, data_path, subid, phase, mode, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.mode = mode
        LABEL_COLUMN = 0
        NAME_COLUMN = 1
        SUB_COLUMN = 2


        df = pd.read_table("dataset/casme2/csame_train.log", sep=',', header=None)
        df.iloc[df.iloc[:, 0] == 2, 0] = 1
        df.iloc[df.iloc[:, 0] == 3, 0] = 2
        df.iloc[df.iloc[:, 0] == 4, 0] = 3


        file_names = df.iloc[:, NAME_COLUMN].values
        label = df.iloc[:, LABEL_COLUMN].values
        # casme:0:Happiness, 1:Repression, 2:Surprise, 3:Disgust,4:Others
        # casme:0:Negative, 1:Positive,2:Surprise
        # samm: 0:Anger, 1:Contempt, 2:Happiness, 3:Surprise,4:Others
        # samm:0:Negative, 1:Positive,2:Surprise
        subject = df.iloc[:, SUB_COLUMN].values
        samplenum=len(subject)

        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Value: {u}, Count: {c}")

        self.file_paths = []
        self.lab = []
        self.sub = []
        if mode == 'supcon':
            if phase == 'train':
                for j in range(0, len(data_path)):
                    if j == 4 or j == 5:
                        for i in range(0, samplenum):
                            if subid != subject[i] and label[i] != 3:  # balance dataset

                                path = os.path.join(self.data_path[j], file_names[i])
                                self.file_paths.append(path)
                                self.lab.append(label[i])
                                self.sub.append(subject[i])
                    else:
                        for i in range(0, samplenum):
                            if subid != subject[i]:
                                path = os.path.join(self.data_path[j], file_names[i])
                                self.file_paths.append(path)
                                self.lab.append(label[i])
                                self.sub.append(subject[i])
        if mode == 'linear':
            if phase == 'train':
                for j in range(0, len(data_path)):
                    if j == 4 or j == 5:  # 放大因子4,5仅用于平衡类别少的类
                        # if j==0 or j==1 or j==2:
                        for i in range(0, samplenum):
                            if subid != subject[i] and label[i] != 4:  # leave one subject out
                                path = os.path.join(self.data_path[j], file_names[i])
                                self.file_paths.append(path)
                                self.lab.append(label[i])
                                self.sub.append(subject[i])
                    else:
                        for i in range(0, samplenum):
                            if subid != subject[i] and label[i] != 4:  # leave one subject out
                                path = os.path.join(self.data_path[j], file_names[i])
                                self.file_paths.append(path)
                                self.lab.append(label[i])
                                self.sub.append(subject[i])
            if phase == 'test':
                for j in range(0, len(data_path)):
                    for i in range(0, samplenum):
                        if subid == subject[i] and label[i] != 4:
                            path = os.path.join(self.data_path[j], file_names[i])
                            self.lab.append(label[i])
                            self.file_paths.append(path)
                            self.sub.append(subject[i])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        if image is None:
            print(path)
        image = image[:, :, ::-1]  # BGR to RGB
        image = image.copy() # H W C
        label = self.lab[idx]
        subject = self.sub[idx]
        if self.transform is not None:
            image = Image.fromarray(image) #CWH
            image = self.transform(image)
        # 查看数据增强后的图像
        #if idx==8:
        # img_tensor = image[0] # CHW
        # plt.figure()
        # img = img_tensor.numpy().transpose((1, 2, 0)) #  HW
        # img = np.clip(img, 0, 1)
        # plt.imshow(img)
        # plt.show()
        # plt.pause(1)
        # plt.close()

        return image, label, subject


def getdata(subid, mode):
    args = parse_args()
    if mode == 'supcon':
        # 加载train data
        data_transforms = transforms.Compose([
             #transforms.RandomResizedCrop(size=224, scale=(0.8, 1.)),#功能：随机长宽裁剪原始照片，最后将照片resize到设定好的size
             transforms.RandomHorizontalFlip(p=0.5),#水平翻转0.2
             transforms.RandomApply([
                 transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
             ], p=0.5),#0.2
             transforms.RandomGrayscale(p=0.2),#转灰度图
             transforms.Resize((224,224)),#HW

             transforms.ToTensor(),
             # transforms.RandomErasing(p=0.2,scale=(0.02, 0.16),ratio =(0.5, 0.5),value = 0,inplace = False)
             # transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                      std=[0.229, 0.224, 0.225])
         ])
        casme2_path=[
            'dataset/casme2/data_FlowNet2/casM0',
                     'dataset/casme2/data_FlowNet2/casM1',
                     'dataset/casme2/data_FlowNet2/casM2',
                     'dataset/casme2/data_FlowNet2/casM3',
                     'dataset/casme2/data_FlowNet2/casM4',
                     'dataset/casme2/data_FlowNet2/casM5',
                     ]

        train_dataset = casme2DataSet(casme2_path, subid, phase='train', mode=mode, transform=TwoCropTransform(data_transforms))
        print('subid:',subid)
        print('Train set size:', train_dataset.__len__())

        # print('data',train_dataset)  #返回的是getitem中的内容，的是图片的tensor，lable，idx
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   num_workers=args.workers,
                                                   shuffle=True,#true
                                                   pin_memory=True)#true
        return train_loader

    else:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # resnet18对应的图片大小是224*224？
            transforms.ToTensor(),
        ])

        casme2_path_train = [
            #  'dataset/casme2/data_FlowNet2/casM0',
            'dataset/casme2/data_FlowNet2/casM1',
            'dataset/casme2/data_FlowNet2/casM2',
            'dataset/casme2/data_FlowNet2/casM3',
            'dataset/casme2/data_FlowNet2/casM4',
            'dataset/casme2/data_FlowNet2/casM5',
        ]  # 放大系数，每个文件中都和csv文件中的数据对应
        casme2_path_test = [
            'dataset/casme2/data_FlowNet2/casM3',
        ]

        train_dataset = casme2DataSet(casme2_path_train, subid, phase='train', transform=data_transforms)
        print('Train set size:', train_dataset.__len__())
        # print('data',train_dataset)  #返回的是getitem中的内容，的是图片的tensor，lable，idx
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=64,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)

        test_dataset = casme2DataSet(casme2_path_test, subid, phase='test',
                                     transform=data_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.workers,
                                                  shuffle=True,
                                                  pin_memory=True)

        print('Test set size:', test_dataset.__len__())

        return train_loader, test_loader









