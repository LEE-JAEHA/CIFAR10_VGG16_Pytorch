"""
Assignment1 for Special Topics of Deep Learning
2019314104_DoGyoon_LEE

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Convolution = [( input - kernel + 2*padding) / stride]+1
#Pooling = [ ( input - dilation* (kernel-1) + 2*padding -1 )   / stride]  + 1
#https://yonghyuc.wordpress.com/2019/08/04/conv-pooling%EC%9D%98-output-size-%EA%B5%AC%ED%95%98%EA%B8%B0/
import math


class MODEL_VGG16_CIFAR10(nn.Module):
    def __init__(self, Temp=1):
        super(MODEL_VGG16_CIFAR10, self).__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),# (1(32-1)- 32 + 3)/2 = 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(2, 2))
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(1, 1),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),stride=(1, 1),padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2),stride=(1, 1)) # modify : changing part / kernel 3->2 if not change when the input 32 X 32 X 3 it returns 512*1 size mismatch
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 10),
        )
        self.Temp = Temp
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logit = self.classifier(x)
        # result = F.softmax(logits, dim=1)

        return logit


