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
    parser.add_argument("--learning_rate",type = float, default=0.01, help="learning rate?")
    parser.add_argument("--batch", type=int, default=8, help="Number of batch?")
    parser.add_argument("--epoch", type=int, default=200, help="Number of Epoch")
    parser.add_argument("--pretrained", default="./checkpoint/",
                        help="file name?")
    parser.add_argument("--mode", type=str, default="train", help="mode select")
    args = parser.parse_args()
    epochs = args.epoch
    print(args) # argument 들 확인

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
    model.to(device) # load to GPU or CPU
    # print(summary(model, (3, 32, 32)))
    # input("TIME")


    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # 학습률 스케줄러는 옵티마이져를 통해 적용된다.
    optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate,weight_decay=1e-4,nesterov=True,momentum=0.9)

    # 원하는 에폭마다, 이전 학습률 대비 변경폭에 따라 학습률을 감소시켜주는 방식
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode='min')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 70], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler`.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc=-1
    best_idx=-1

    best_accuracy_test=-1;best_accuracy_train=-1
    train_loss = list();test_loss = list();train_accuracy = list();test_accuracy = list();

    for epoch in range(1, epochs + 1):
        print("Epoch {0} : ".format(epoch))
        running_loss = 0
        # scheduler.step()
        for epoch in range(1, epochs + 1):
            print("Epoch {0} : ".format(epoch))
            sum_loss = 0
            for idx, data in enumerate(tqdm(data_.train_loader), 1):  # 한번의 for문 마다 전체 데이터를 나눈만큼 돈다
                model.train()
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)

                pred = model(imgs)
                loss = criterion(pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            print("Loss of train : {0}".format(sum_loss))
            train_loss.append(sum_loss)

        model.eval()
        with torch.no_grad():
            sum_loss = 0
            correct = 0
            for idx, data in enumerate(tqdm(data_.testloader), 1):
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(imgs)
                pred = model(imgs)
                loss = criterion(pred, labels)
                sum_loss += loss.item()
                _, pred_hot_label = torch.max(pred, 1)
                correct += pred_hot_label.eq(labels).sum().item()

        acc = 100 * correct/len(data_.testloader)
        if acc > best_acc:
            print("SAVING MODEL")
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            chkpoint_file=args.file_name
            if not os.path.isdir(chkpoint_file):
                os.mkdir(chkpoint_file)
            path_ = "./"+chkpoint_file+"/"
            file_list = os.listdir(path_)
            if len(file_list)!=0:
                for i in file_list:
                    os.remove(path_+i)
            print("Accuracy of this model : {}".format(acc))
            b = str(acc);
            b = b[:5];
            b = b.replace(".", "_")
            net_teacher_name = "./" + chkpoint_file + "/epoch" + str(epoch) + "_acc" + str(b) + ".pth"
            print(net_teacher_name)
            model_name = "./"+chkpoint_file+"/epoch"+str(epoch)+"_acc"+str(b)+".pth"
            torch.save(state, model_name)
            best_acc = acc
            best_idx = epoch
        print("*" * 10)
        scheduler.step() # Optimizer만 사용하려면 얘 지우고 optimizer 부분 주석 제거

    print("BEST ACC : {0}% / best_epoch : {1} ".format(best_acc, best_idx + 1))
    print("Best Test  accuracy : {0}".format(best_acc))
    x = [i for i in range(0, epochs, 1)]
    plt.plot(x, train_loss, label="Train")
    plt.title("Loss of Train set")
    plt.legend()
    file_ = str(epochs) + "_Loss_" + args.file_name + ".png"
    save_ = "./"+args.file_name+"/"+file_
    plt.savefig(save_, dpi=300)

    plt.clf()
    x = [i for i in range(0, epochs, 1)]
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(x, test_accuracy, label="Test")
    plt.title("Accuracy of Test set\nTest_acc: {0}".format(best_acc))
    plt.legend()
    file_ = str(epochs) + "_Accuracy_" + args.file_name + "_test_"+ str(best_acc) +"_train_" + ".png"
    save_ = "./"+args.file_name+"/"+file_
    plt.savefig(save_, dpi=300)
    print(args)
#####
if __name__ == "__main__" :
    train()
