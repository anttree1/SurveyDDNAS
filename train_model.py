import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from scipy.io import loadmat
from torch.utils.data.dataset import random_split
import numpy as np
import argparse
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import time
import sys
import model_class as mc

parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集和测试集比例')
parser.add_argument('--epochs', type=int, default=50, help='训练轮次')
parser.add_argument('--lr', type=float, default=0.1, help='学习率')
parser.add_argument('--momentum', type=float, default=0.9, help='动量')
parser.add_argument('--batch_size', type=int, default=2, help='批量大小')
parser.add_argument('--drop_size', type=float, default=0.2, help='drop大小')
parser.add_argument('--load_path', type=str, default='F:\data\surveyeeg\sch1\Schizophrenia_1.mat', help='载入数据地址')
args = parser.parse_args()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
log_file = f"F:/=WORK=/Work-survey1/logging/VGG_sch1/output_{timestamp}.log"
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.info(f'train_ratio: {args.train_ratio:.4f}')
logging.info(f'epochs: {args.epochs:.4f}')
logging.info(f'lr: {args.lr:.4f}')
logging.info(f'momentum: {args.momentum:.4f}')
logging.info(f'batch_size: {args.batch_size:.4f}')
logging.info(f'drop_size: {args.drop_size:.4f}')


data = loadmat(args.load_path)
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
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
classes = ('0', '1')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # model = mc.MLP(args.drop_size)
    # model = mc.VGG(args.drop_size)
    # model = mc.ResNet(mc.ResidualBlock, [2, 2, 2, 2], num_classes=2)
    model = mc.BPR()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train(device, optimizer, model, criterion, args.epochs)
    test(device, model)


def train(device, optimizer, model, criterion, epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.float()
            targets = torch.tensor(targets).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(trainloader):.4f}')
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(trainloader):.4f}')


def test(device, model):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            inputs = inputs.float()
            targets = torch.tensor(targets).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    # 计算分类指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    logging.info('------')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')


if __name__ == '__main__':
    main()
