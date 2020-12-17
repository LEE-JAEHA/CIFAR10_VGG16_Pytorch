from model.vgg16 import MODEL_VGG16_CIFAR10
from datasets.cifar10 import  CIFAR10
import torch
import os
import argparse
from tqdm import tqdm # data 진행 속도 확인
from torchsummary import summary # model 구조 확인
import matplotlib.pyplot as plt
def train():
    parser = argparse.ArgumentParser(description="helper")
    parser.add_argument("--save_model", default="./checkpoint", help="save")
    parser.add_argument("--save_plot", default="./checkpoint", help="save")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate?")
    parser.add_argument("--batch", type=int, default=8, help="Number of batch?")
    parser.add_argument("--epoch", type=int, default=200, help="Number of Epoch")
    parser.add_argument("--pretrained", default="./checkpoint/",
                        help="file name?")
    parser.add_argument("--mode", type=str, default="train", help="mode select")
    args = parser.parse_args()
    epochs = args.epoch
    print(args)  # argument 들 확인

    print(args.batch)

    data_ = CIFAR10(batch=args.batch)
    #GPU 확인
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())

    if torch.cuda.is_available():
        device = torch.device("cuda:0"); print("using cuda:0")
    else:
        device = torch.device("cpu") ; print("using CPU")
    print("Device ? : {0}".format(device))


    model = MODEL_VGG16_CIFAR10()  # load model
    # for i,v in model.named_parameters():
    #     print(i)
    # input("TIME")


    # model = MODEL_QEUSTION2_WITHFC512()  # load model
    # print(summary(model, (3, 32, 32)))
    # input("TIME")

    last_layer = ['classifier.6.weight', 'classifier.6.bias']
    # retrain_layer_list = ['classifier.6.weight', 'classifier.6.bias',
    #                       'classifier.3.weight', 'classifier.3.bias',
    #                       ]

    retrain_layer_list = ['classifier.6.weight', 'classifier.6.bias',
                          'classifier.0.weight', 'classifier.0.bias',
                          'classifier.3.weight', 'classifier.3.bias',
                          'block_5.6.weight', 'block_5_6.bias',
                          'block_5.3.weight', 'block_5_3.bias',
                        'block_5.0.weight', 'block_5_0.bias',
                          ]
    pretrained_dict = torch.load(pretrained)['net']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in  last_layer}
    # for i in pretrained_dict.keys():
    #     print(i)
    # input("TIME")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


    for name,para in model.named_parameters():
        if name in retrain_layer_list:
            para.requires_grad = True
        else:
            para.requires_grad = False
    print(retrain_layer_list)
    input("Load Finish Start? : ")

    # if os.path.exists(pretrained):
    #     print("==> load checkpoint")
    #     checkpoint = torch.load(pretrained)
    #     model.load_state_dict(checkpoint['net'])
    #     print("==> model load success")
    # else:
    #     print("There is no pth file or root error.\nProgram End")
    #     exit(0)


