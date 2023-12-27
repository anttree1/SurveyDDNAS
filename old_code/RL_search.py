import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from old_code.controller import Controller
from model_search import SharedCNN
from utils import AverageMeter
from torch.utils.data.dataset import random_split
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='ENAS')

parser.add_argument('--search_for', default='macro', choices=['macro'])
parser.add_argument('--data_path', default='../data/', type=str)
parser.add_argument('--output_filename', default='ENAS', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--eval_every_epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fixed_arc', action='store_true', default=False)

parser.add_argument('--child_num_layers', type=int, default=7)
parser.add_argument('--child_out_filters', type=int, default=36)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_l2_reg', type=float, default=0.00025)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_keep_prob', type=float, default=0.9)
parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T', type=float, default=10)

parser.add_argument('--controller_lstm_size', type=int, default=36)
parser.add_argument('--controller_lstm_num_layers', type=int, default=1)
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001)
parser.add_argument('--controller_train_every', type=int, default=1)
parser.add_argument('--controller_num_aggregate', type=int, default=20)
parser.add_argument('--controller_train_steps', type=int, default=10)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_tanh_constant', type=float, default=1.5)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)
parser.add_argument('--controller_skip_target', type=float, default=0.4)
parser.add_argument('--controller_skip_weight', type=float, default=0.8)
parser.add_argument('--controller_bl_dec', type=float, default=0.99)
parser.add_argument('--load_path', type=str, default='/home/ai/data/wangxingy/data/survey/sch1/Sch_1.mat')
parser.add_argument('--train_ratio', type=float, default=0.6, help='训练集和测试集比例')

args = parser.parse_args()


def load_datasets():
    """Create data loaders for the CIFAR-10 dataset.

    Returns: Dict containing data loaders.
    """
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root=args.data_path,
                                     train=True,
                                     transform=valid_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=args.data_path,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    train_indices = list(range(0, 45000))
    valid_indices = list(range(45000, 50000))
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)

    data_loaders = {}
    data_loaders['train_subset'] = torch.utils.data.DataLoader(dataset=train_subset,
                                                               batch_size=args.batch_size,
                                                               shuffle=True,
                                                               pin_memory=True,
                                                               num_workers=2)

    data_loaders['valid_subset'] = torch.utils.data.DataLoader(dataset=valid_subset,
                                                               batch_size=args.batch_size,
                                                               shuffle=True,
                                                               pin_memory=True,
                                                               num_workers=2,
                                                               drop_last=True)

    data_loaders['train_dataset'] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                batch_size=args.batch_size,
                                                                shuffle=True,
                                                                pin_memory=True,
                                                                num_workers=2)

    data_loaders['test_dataset'] = torch.utils.data.DataLoader(dataset=test_dataset,
                                                               batch_size=args.batch_size,
                                                               shuffle=False,
                                                               pin_memory=True,
                                                               num_workers=2)

    return data_loaders


def train_shared_cnn(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     shared_cnn_optimizer,
                     fixed_arc=None):
    global vis_win

    controller.eval()

    if fixed_arc is None:
        # Use a subset of the training set when searching for an arhcitecture
        train_loader = data_loaders['train_subset']
    else:
        # Use the full training set when training a fixed architecture
        train_loader = data_loaders['train_dataset']

    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    for i, (images, labels) in enumerate(train_loader):
        start = time.time()
        images = images.cuda()
        labels = labels.cuda()
        images = images.float()
        labels = torch.tensor(labels).long()

        if fixed_arc is None:
            with torch.no_grad():
                controller()  # perform forward pass to generate a new architecture
            sample_arc = controller.sample_arc
        else:
            sample_arc = fixed_arc

        shared_cnn.zero_grad()
        pred = shared_cnn(images, sample_arc)
        loss = nn.CrossEntropyLoss()(pred, labels)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), args.child_grad_bound)
        shared_cnn_optimizer.step()

        train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))

        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())

        end = time.time()

        if (i) % args.log_every == 0:
            learning_rate = shared_cnn_optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tch_step=' + str(i) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tlr=%.4f' % (learning_rate) + \
                      '\t|g|=%.4f' % (grad_norm.item()) + \
                      '\tacc=%.4f' % (train_acc_meter.val) + \
                      '\ttime=%.2fit/s' % (1. / (end - start))
            print(display)

    controller.train()


def train_controller(epoch,
                     controller,
                     shared_cnn,
                     data_loaders,
                     controller_optimizer,
                     baseline=None):
    print('Epoch ' + str(epoch) + ': Training controller')

    global vis_win

    shared_cnn.eval()
    valid_loader = data_loaders['valid_subset']

    reward_meter = AverageMeter()
    baseline_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    controller.zero_grad()
    for i in range(args.controller_train_steps * args.controller_num_aggregate):
        start = time.time()
        images, labels = next(iter(valid_loader))
        images = images.cuda()
        labels = labels.cuda()
        images = images.float()
        labels = torch.tensor(labels).long()
        controller()
        sample_arc = controller.sample_arc

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        val_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))

        reward = torch.tensor(val_acc.detach())
        reward += args.controller_entropy_weight * controller.sample_entropy

        if baseline is None:
            baseline = val_acc
        else:
            baseline -= (1 - args.controller_bl_dec) * (baseline - reward)
            # detach to make sure that gradients are not backpropped through the baseline
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

        if (i + 1) % args.controller_num_aggregate == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.child_grad_bound)
            controller_optimizer.step()
            controller.zero_grad()

            if (i + 1) % (2 * args.controller_num_aggregate) == 0:
                learning_rate = controller_optimizer.param_groups[0]['lr']
                display = 'ctrl_step=' + str(i // args.controller_num_aggregate) + \
                          '\tloss=%.3f' % (loss_meter.val) + \
                          '\tent=%.2f' % (controller.sample_entropy.item()) + \
                          '\tlr=%.4f' % (learning_rate) + \
                          '\t|g|=%.4f' % (grad_norm.item()) + \
                          '\tacc=%.4f' % (val_acc_meter.val) + \
                          '\tbl=%.2f' % (baseline_meter.val) + \
                          '\ttime=%.2fit/s' % (1. / (end - start))
                print(display)

    shared_cnn.train()
    return baseline


def evaluate_model(epoch, controller, shared_cnn, data_loaders, n_samples=10):
    controller.eval()
    shared_cnn.eval()

    print('Here are ' + str(n_samples) + ' architectures:')
    best_arc, _ = get_best_arc(controller, shared_cnn, data_loaders, n_samples, verbose=True)

    valid_loader = data_loaders['valid_subset']
    test_loader = data_loaders['test_subset']

    valid_acc = get_eval_accuracy(valid_loader, shared_cnn, best_arc)
    test_acc = get_eval_accuracy(test_loader, shared_cnn, best_arc)

    print('Epoch ' + str(epoch) + ': Eval')
    print('valid_accuracy: %.4f' % (valid_acc))
    print('test_accuracy: %.4f' % (test_acc))

    controller.train()
    shared_cnn.train()


def get_best_arc(controller, shared_cnn, data_loaders, n_samples=10, verbose=False):
    controller.eval()
    shared_cnn.eval()

    valid_loader = data_loaders['valid_subset']

    images, labels = next(iter(valid_loader))
    images = images.cuda()
    labels = labels.cuda()
    images = images.float()
    labels = torch.tensor(labels).long()
    arcs = []
    val_accs = []
    for i in range(n_samples):
        with torch.no_grad():
            controller()  # perform forward pass to generate a new architecture
        sample_arc = controller.sample_arc
        arcs.append(sample_arc)

        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        val_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        val_accs.append(val_acc.item())

        if verbose:
            print_arc(sample_arc)
            print('val_acc=' + str(val_acc.item()))
            print('-' * 80)

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]

    controller.train()
    shared_cnn.train()
    return best_arc, best_val_acc


def get_eval_accuracy(loader, shared_cnn, sample_arc):
    total = 0.
    acc_sum = 0.
    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()
        images = images.float()
        labels = torch.tensor(labels).long()
        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        acc_sum += torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
        total += pred.shape[0]

    acc = acc_sum / total
    return acc.item()


def print_arc(sample_arc):
    for key, value in sample_arc.items():
        if len(value) == 1:
            branch_type = value[0].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in branch_type) + ']')
        else:
            branch_type = value[0].cpu().numpy().tolist()
            skips = value[1].cpu().numpy().tolist()
            print('[' + ' '.join(str(n) for n in (branch_type + skips)) + ']')


def train_enas(start_epoch,
               controller,
               shared_cnn,
               data_loaders,
               shared_cnn_optimizer,
               controller_optimizer,
               shared_cnn_scheduler):
    baseline = None
    for epoch in range(start_epoch, args.num_epochs):

        train_shared_cnn(epoch,
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

        if epoch % args.eval_every_epochs == 0:
            evaluate_model(epoch, controller, shared_cnn, data_loaders)

        shared_cnn_scheduler.step(epoch)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'shared_cnn_state_dict': shared_cnn.state_dict(),
                 'controller_state_dict': controller.state_dict(),
                 'shared_cnn_optimizer': shared_cnn_optimizer.state_dict(),
                 'controller_optimizer': controller_optimizer.state_dict()}
        filename = 'checkpoints/' + args.output_filename + '.pth.tar'
        torch.save(state, filename)


def train_fixed(start_epoch,
                controller,
                shared_cnn,
                data_loaders):
    best_arc, best_val_acc = get_best_arc(controller, shared_cnn, data_loaders, n_samples=100, verbose=False)
    print('Best architecture:')
    print_arc(best_arc)
    print('Validation accuracy: ' + str(best_val_acc))

    fixed_cnn = SharedCNN(num_layers=args.child_num_layers,
                          num_branches=args.child_num_branches,
                          out_filters=512 // 4,  # args.child_out_filters
                          keep_prob=args.child_keep_prob,
                          fixed_arc=best_arc)
    fixed_cnn = fixed_cnn.cuda()

    fixed_cnn_optimizer = torch.optim.SGD(params=fixed_cnn.parameters(),
                                          lr=args.child_lr_max,
                                          momentum=0.9,
                                          nesterov=True,
                                          weight_decay=args.child_l2_reg)

    fixed_cnn_scheduler = CosineAnnealingLR(optimizer=fixed_cnn_optimizer,
                                            T_max=args.child_lr_T,
                                            eta_min=args.child_lr_min)

    test_loader = data_loaders['test_dataset']

    for epoch in range(args.num_epochs):

        train_shared_cnn(epoch,
                         controller,  # not actually used in training the fixed_cnn
                         fixed_cnn,
                         data_loaders,
                         fixed_cnn_optimizer,
                         best_arc)

        if epoch % args.eval_every_epochs == 0:
            test_acc = get_eval_accuracy(test_loader, fixed_cnn, best_arc)
            print('Epoch ' + str(epoch) + ': Eval')
            print('test_accuracy: %.4f' % (test_acc))

        fixed_cnn_scheduler.step(epoch)

        state = {'epoch': epoch + 1,
                 'args': args,
                 'best_arc': best_arc,
                 'fixed_cnn_state_dict': shared_cnn.state_dict(),
                 'fixed_cnn_optimizer': fixed_cnn_optimizer.state_dict()}
        filename = 'checkpoints/' + args.output_filename + '_fixed.pth.tar'
        torch.save(state, filename)


def main():
    global args
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(args)

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
    classes = ('0', '1')

    controller = Controller(num_layers=args.child_num_layers,
                            num_branches=args.child_num_branches,
                            lstm_size=args.controller_lstm_size,
                            lstm_num_layers=args.controller_lstm_num_layers,
                            tanh_constant=args.controller_tanh_constant,
                            temperature=None,
                            skip_target=args.controller_skip_target,
                            skip_weight=args.controller_skip_weight)
    controller = controller.cuda()

    shared_cnn = SharedCNN(num_layers=args.child_num_layers,
                           num_branches=args.child_num_branches,
                           out_filters=args.child_out_filters,
                           keep_prob=args.child_keep_prob)
    shared_cnn = shared_cnn.cuda()

    controller_optimizer = torch.optim.Adam(params=controller.parameters(),
                                            lr=args.controller_lr,
                                            betas=(0.0, 0.999),
                                            eps=1e-3)

    shared_cnn_optimizer = torch.optim.SGD(params=shared_cnn.parameters(),
                                           lr=args.child_lr_max,
                                           momentum=0.9,
                                           nesterov=True,
                                           weight_decay=args.child_l2_reg)

    shared_cnn_scheduler = CosineAnnealingLR(optimizer=shared_cnn_optimizer,
                                             T_max=args.child_lr_T,
                                             eta_min=args.child_lr_min)

    start_epoch = 0

    train_enas(start_epoch,
               controller,
               shared_cnn,
               data_loaders,
               shared_cnn_optimizer,
               controller_optimizer,
               shared_cnn_scheduler)


if __name__ == "__main__":
    main()
