import torch
from torch.utils.data.dataset import random_split
from scipy.io import loadmat
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
import model_4 as m4
import utils

parser = argparse.ArgumentParser(description='surveyDDNAS')
parser.add_argument('--path_sch1', type=str, default='/home/ai/data/wangxingy/data/survey/sch1/Sch_11.mat',
                    help='小数据集,(84,30,16,256)')
parser.add_argument('--path_ep1', type=str, default='/home/ai/data/wangxingy/data/survey/ep1/Ep_2.mat',
                    help='大数据集,(677,10,55,500)')
parser.add_argument('--path_pa1', type=str, default='/home/ai/data/wangxingy/data/survey/pa1/Pa_1.mat',
                    help='噪声数据集,(840,4,63,500)')

parser.add_argument('--inp_size', type=int, default=10, help='sch=30,pa=4,ep=10')
parser.add_argument('--drop_size', type=float, default=0.5, help='dropout参数')
parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集和测试集比例')
parser.add_argument('--batch_size', type=int, default=8, help='批量大小')
parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='权重衰减')
parser.add_argument('--epochs', type=int, default=200, help='训练轮次')
parser.add_argument('--step_size', type=int, default=500, help='衰减轮次')
parser.add_argument('--gamma', type=float, default=0.8, help='衰减系数')
args = parser.parse_args()


def data_load(load_path):
    data = loadmat(load_path)
    # 获取数据和标签
    inputs = data['data']
    labels = data['label']
    labels = labels.squeeze(0)
    # 定义训练集和测试集的比例（例如，80%的数据用于训练，20%的数据用于测试）
    train_ratio = args.train_ratio
    test_ratio = 1 - train_ratio
    # 计算划分的样本数量
    train_size = int(train_ratio * len(inputs))
    test_size = len(inputs) - train_size
    # 使用random_split函数划分索引
    train_indices, test_indices = random_split(range(len(inputs)), [train_size, test_size])
    # 根据索引划分数据集和标签
    train_dataset = torch.utils.data.Subset(inputs, train_indices)
    test_dataset = torch.utils.data.Subset(inputs, test_indices)
    # 将Subset对象转换为NumPy数组
    train_data = np.array([inputs[i] for i in train_dataset.indices])
    test_data = np.array([inputs[i] for i in test_dataset.indices])
    train_labels = np.array([labels[i] for i in train_indices])
    test_labels = np.array([labels[i] for i in test_indices])
    trainset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    testset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    data_loaders = {}
    data_loaders['train_subset'] = trainloader
    data_loaders['valid_subset'] = validloader
    data_loaders['test_subset'] = testloader
    return data_loaders


def get_arc(arc_list):
    search_arc = {}
    for i, var_layer in enumerate(arc_list):
        search_arc[str(i)] = [var_layer[0]]
        search_arc[str(i)].append(torch.tensor(var_layer[1]))
        search_arc[str(i)].append(torch.tensor(var_layer[2]))
        search_arc[str(i)].append(torch.tensor(var_layer[3]))
    return search_arc


def main():
    arch_4 = [[2, 1, 2, 1], [3, 2, 3, 2], [2, 2, 0, 0], [3, 2, 2, 3], [1, 3, 1, 4]]
    search_arc = get_arc(arch_4)
    model = m4.SharedCNN(drop_size=args.drop_size, inp_size=args.inp_size)
    # dataloader = data_load(args.path_sch1)
    # dataloader = data_load(args.path_pa1)
    dataloader = data_load(args.path_ep1)

    trainloader = dataloader['train_subset']
    testloader = dataloader['test_subset']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
                          )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(args.epochs):
        print('epoch:', epoch)
        train(device, optimizer, model, criterion, trainloader, search_arc)
        test(device, model, testloader, search_arc, criterion)
        scheduler.step()


def train(device, optimizer, model, criterion, trainloader, search_arc):
    model.train()
    x_true = []
    x_pred = []
    running_loss = 0.0
    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.float()
        targets = targets.clone().detach().long()
        optimizer.zero_grad()
        outputs = model(inputs, search_arc)
        loss = criterion(outputs, targets)
        _, logits = torch.max(outputs.data, 1)
        x_true.extend(targets.cpu().numpy())
        x_pred.extend(logits.cpu().numpy())
        loss.backward()
        optimizer.step()
        accuracy = accuracy_score(x_true, x_pred)
        running_loss += loss.item()
    print('train_loss:', running_loss / len(trainloader), 'train_acc:', accuracy)


def test(device, model, testloader, search_arc, criterion):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.float()
            targets = targets.clone().detach().long()
            outputs = model(inputs, search_arc)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            running_loss += loss.item()
    # 计算分类指标
    print('true:', y_true)
    print('pred:', y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    if tn + fp == 0:
        specificity = 'nan'
    else:
        specificity = tn / (tn + fp)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print('Test_loss:', running_loss / len(testloader), 'Test_Acc:', accuracy, 'Test_Spe:', specificity, 'Test_Sen:',
          recall, 'Test_F1:', f1)


if __name__ == '__main__':
    main()
