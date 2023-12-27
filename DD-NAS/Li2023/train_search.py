import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from controller import Controller
from model_search import SharedCNN
from utils import *
from torch.utils.data.dataset import random_split
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Li2023')

parser.add_argument('--child_num_layers', type=int, default=5, help='搜索到的CNN网络的层数')
parser.add_argument('--child_l2_reg', type=float, default=0.00025, help='L2正则化系数')
parser.add_argument('--child_num_branches', type=int, default=5, help='候选操作数')
parser.add_argument('--child_lr_max', type=float, default=0.05, help='CNN学习率')
parser.add_argument('--child_lr_min', type=float, default=0.0005, help='CNN微调学习率的最小值')
parser.add_argument('--child_lr_T', type=float, default=10, help='CNN微调学习率的迭代次数')
parser.add_argument('--child_train_epoch', type=int, default=50, help='CNN微调学习率的迭代次数')

parser.add_argument('--controller_lstm_size', type=int, default=36, help='LSTM输入维度')
parser.add_argument('--controller_lstm_num_layers', type=int, default=1, help='LSTM层数')
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001, help='熵的权重')
parser.add_argument('--controller_num_aggregate', type=int, default=20, help='控制器数量')
parser.add_argument('--controller_train_steps', type=int, default=10, help='控制器的训练轮次')
parser.add_argument('--controller_lr', type=float, default=0.001, help='控制器的学习率')
parser.add_argument('--controller_tanh_constant', type=float, default=1.5, help='tanh参数')
parser.add_argument('--controller_skip_target', type=float, default=0.4, help='跨层连接的参数')
parser.add_argument('--controller_skip_weight', type=float, default=0.8, help='跨层连接的权重')
parser.add_argument('--controller_bl_dec', type=float, default=0.99, help='baseline dec')

parser.add_argument('--batch_size', type=int, default=4, help='batch大小')
parser.add_argument('--num_epochs', type=int, default=10, help='训练总轮次')
parser.add_argument('--seed', type=int, default=0, help='随机种子')
parser.add_argument('--drop_size', type=float, default=0.1, help='dropout参数')
parser.add_argument('--load_path', type=str, default='/home/ai/data/wangxingy/data/survey/sch1/Sch_1.mat', help='数据集地址')
parser.add_argument('--train_ratio', type=float, default=0.6, help='训练集和测试集比例')


args = parser.parse_args()


# 加载数据集
def data_load():
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
    validloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    data_loaders = {}
    data_loaders['train_subset'] = trainloader
    data_loaders['valid_subset'] = validloader
    data_loaders['test_subset'] = testloader
    return data_loaders


def train_shared_cnn(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     shared_cnn_optimizer,
                     ):
    global vis_win  # 声明全局变量
    print('Epoch ' + str(epoch) + ': Training CNN')  # 打印当前训练控制器的epoch
    # 将控制器模型设置为评估模式，在前向传播期间不会跟踪梯度
    controller.eval()

    train_loader = data_loaders['train_subset']

    # 用于计算训练精度和损失的平均值
    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    for train_epoch in range(args.child_train_epoch):
        for i, (images, labels) in enumerate(train_loader):
            start = time.time()
            images = images.cuda()
            labels = labels.cuda()
            images = images.float()
            labels = labels.long().clone().detach()

            # 使用控制器模型进行前向传播以生成一个新的架构
            with torch.no_grad():
                controller()
            sample_arc = controller.sample_arc  # 新架构 sample_arc

            shared_cnn.zero_grad()  # 梯度清零
            pred = shared_cnn(images, sample_arc)
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()
            shared_cnn_optimizer.step()
            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            train_acc_meter.update(train_acc.item())
            loss_meter.update(loss.item())
            end = time.time()

        display = 'epoch=' + str(train_epoch) + \
                  '\tloss=%.6f' % (loss_meter.val) + \
                  '\tacc=%.4f' % (train_acc_meter.val) + \
                  '\ttime=%.2fit/s' % (1. / (end - start))
        print(display)

    # 将控制器模型重新设置为训练模式
    controller.train()


def train_controller(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     controller_optimizer,
                     baseline=None):
    print('Epoch ' + str(epoch) + ': Training controller')  # 打印当前训练控制器的epoch

    global vis_win

    # 调整CNN为评估模式，不传递梯度
    shared_cnn.eval()
    valid_loader = data_loaders['valid_subset']  # 获取验证集

    reward_meter = AverageMeter()
    baseline_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    controller.zero_grad()  # 梯度清零

    # 迭代控制器训练的步骤数乘以控制器的数目
    for i in range(args.controller_train_steps * args.controller_num_aggregate):
        start = time.time()
        images, labels = next(iter(valid_loader))
        images = images.cuda()
        labels = labels.cuda()
        images = images.float()
        labels = labels.long().clone().detach()
        controller()
        sample_arc = controller.sample_arc

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        val_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))

        reward = val_acc.clone().detach()
        reward += args.controller_entropy_weight * controller.sample_entropy

        if baseline is None:
            baseline = val_acc
        else:
            baseline -= (1 - args.controller_bl_dec) * (baseline - reward)
            baseline = baseline.detach()

        loss = -1 * controller.sample_log_prob * (reward - baseline)

        if args.controller_skip_weight is not None:
            loss += args.controller_skip_weight * controller.skip_penaltys

        reward_meter.update(reward.item())
        baseline_meter.update(baseline.item())
        val_acc_meter.update(val_acc.item())
        loss_meter.update(loss.item())
        loss = loss / args.controller_num_aggregate
        loss.backward(retain_graph=True)
        end = time.time()

        # 每经过args.controller_num_aggregate步，进行一次梯度裁剪、参数更新和输出调试信息
        if (i + 1) % args.controller_num_aggregate == 0:
            controller_optimizer.step()
            controller.zero_grad()

    display = '\tloss=%.3f' % (loss_meter.val) + \
              '\tent=%.2f' % (controller.sample_entropy.item()) + \
              '\tacc=%.4f' % (val_acc_meter.val) + \
              '\ttime=%.2fit/s' % (1. / (end - start))
    print(display)

    shared_cnn.train()
    return baseline


def evaluate_model(epoch, controller, shared_cnn, data_loaders, n_samples=1):
    controller.eval()
    shared_cnn.eval()

    print('Searched architectures:')
    best_arc, _ = get_best_arc(controller, shared_cnn, data_loaders, n_samples, verbose=True)

    valid_loader = data_loaders['valid_subset']
    test_loader = data_loaders['test_subset']

    valid_acc = get_eval_accuracy(valid_loader, shared_cnn, best_arc)
    test_acc = get_eval_accuracy(test_loader, shared_cnn, best_arc)

    print('Epoch ' + str(epoch) + ': Eval')
    print('valid_accuracy: %.4f' % (valid_acc))
    print('test_accuracy: %.4f' % (test_acc))
    print('-' * 80)

    controller.train()
    shared_cnn.train()


def train_enas(start_epoch,
               controller,
               shared_cnn,
               data_loaders,
               shared_cnn_optimizer,
               controller_optimizer,
               shared_cnn_scheduler):
    baseline = None  # 用于存储控制器策略的奖励基准值
    for epoch in range(start_epoch, args.num_epochs):
        train_shared_cnn(epoch,  # 当前训练轮次
                         controller,
                         shared_cnn,
                         data_loaders,
                         shared_cnn_optimizer)

        baseline = train_controller(epoch,
                                    controller,
                                    shared_cnn,
                                    data_loaders,
                                    controller_optimizer,
                                    baseline)

        # 训练一定周期时，调用 evaluate_model 函数对当前模型进行评估。
        evaluate_model(epoch, controller, shared_cnn, data_loaders)

        # 使用共享卷积神经网络模型的学习率调度器 shared_cnn_scheduler
        # 通过调用 step 方法来更新学习率
        shared_cnn_scheduler.step(epoch)


def main():
    global args  # 声明args是全局变量
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(args)

    data_loaders = data_load()
    controller = Controller(num_layers=args.child_num_layers,  # 搜索到的网络的层数
                            num_branches=args.child_num_branches,  # 候选操作数
                            lstm_size=args.controller_lstm_size,  # LSTM输入维度
                            lstm_num_layers=args.controller_lstm_num_layers,  # LSTM层数
                            tanh_constant=args.controller_tanh_constant,  # 正切参数，用于调整logits的大小
                            temperature=None,  # 另一个调整参数
                            skip_target=args.controller_skip_target,  # 控制跳跃连接出现的比例
                            skip_weight=args.controller_skip_weight)
    controller = controller.cuda()

    shared_cnn = SharedCNN(drop_size=args.drop_size)
    shared_cnn = shared_cnn.cuda()

    controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                            lr=args.controller_lr,  # 控制器的学习率
                                            betas=(0.0, 0.999),  # 指数衰减因子
                                            eps=1e-3  # 用于数值稳定性的小常数
                                            )

    shared_cnn_optimizer = torch.optim.SGD(params=shared_cnn.parameters(),
                                           lr=args.child_lr_max,  # CNN的学习率
                                           momentum=0.9,  # 动量
                                           nesterov=True,  # 用于加速收敛并改善梯度跟踪的能力的布尔值
                                           weight_decay=args.child_l2_reg  # L2正则化系数
                                           )

    # CosineAnnealingLR 是 PyTorch 提供的学习率调度器之一，它用于在训练过程中动态地调整学习率。
    # 方法基于余弦函数的形式，在训练过程中按照余弦曲线调整学习率。具体来说，学习率在一个周期内从最大值衰减到最小值，然后重新回升到最大值，再循环下去。
    # 该方法具有改善模型鲁棒性，解决学习率大小不合理，降低训练后期震荡的作用
    shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                             T_max=args.child_lr_T,  # 一个周期的迭代次数。默认为 100
                                             eta_min=args.child_lr_min  # 学习率的最小值
                                             )

    start_epoch = 0

    train_enas(start_epoch,  # 开始轮次
               controller,  # 控制器
               shared_cnn,  # CNN
               data_loaders,  # 加载的数据
               shared_cnn_optimizer,  # CNN的优化器
               controller_optimizer,  # 控制器的优化器
               shared_cnn_scheduler  # CNN的学习率调度器
               )


if __name__ == "__main__":
    main()
