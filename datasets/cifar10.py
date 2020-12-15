import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
"""torchvision datasets은 pickle module로 객체 구조를 갖는 cifar10 데이터들을 serialize,
    pickle 설명> https://pythontips.com/2013/08/02/what-is-pickle-in-python/
    pillow library로 해당 데이터를 이미지화합니다.  
    데이터를 그냥 불러도 되지만, 대부분 데이터를 transform하죠! 그 이유는? Batch Normalization을
    생각하면 편할 것 같아요 :) 
     https://discuss.pytorch.org/t/understanding-transform-normalize/21730/2
     """

class CIFAR10():
    def __init__(self,batch=8):
        self.batch_size = batch
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform2 =transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                # transforms.RandomAffine(
                #     degrees=(10, 30),
                #     translate=(0.25, 0.5),
                #     scale=(1.2, 2.0),
                # ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform2)
        self.validset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                     download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=self.transform)
        
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size,
                                                  shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False, num_workers=2)


"""CIFAR 10의 클라스는 총 10개로 다음과 같습니다"""

# import matplotlib.pyplot as plt
# import numpy as np
#
# # functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize < 변환 과정에서 normalize를 해줬으니 변환해줘야겠죠.
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))