import torch
import numpy as np


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


def print_arc(sample_arc):
    list_out = []
    for key, value in sample_arc.items():
        list1 = []
        list1.append(int(key))
        list1.append(value[0])
        list1.append(value[1].item())
        list1.append(value[2].item())
        list_out.append(list1)
    print(list_out)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_eval_accuracy(loader, shared_cnn, sample_arc):
    total = 0.
    acc_sum = 0.
    for (images, labels) in loader:
        images = images.cuda()
        labels = labels.cuda()
        images = images.float()
        labels = labels.long().clone().detach()
        with torch.no_grad():
            pred = shared_cnn(images, sample_arc)
        acc_sum += torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
        total += pred.shape[0]

    acc = acc_sum / total
    return acc.item()


def get_best_arc(controller, shared_cnn, data_loaders, n_samples=10, verbose=False):
    controller.eval()
    shared_cnn.eval()

    valid_loader = data_loaders['valid_subset']

    images, labels = next(iter(valid_loader))
    images = images.cuda()
    labels = labels.cuda()
    images = images.float()
    labels = labels.long().clone().detach()
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

    best_iter = np.argmax(val_accs)
    best_arc = arcs[best_iter]
    best_val_acc = val_accs[best_iter]

    controller.train()
    shared_cnn.train()
    return best_arc, best_val_acc
