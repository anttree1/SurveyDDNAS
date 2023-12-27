import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FactorizedReduction(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(FactorizedReduction, self).__init__()

        assert out_planes % 2 == 0, (
            "Need even number of filters when using this factorized reduction.")

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        if stride == 1:
            self.fr = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False))
        else:
            self.path1 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))

            self.path2 = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                nn.Conv2d(in_planes, out_planes // 2, kernel_size=1, bias=False))
            self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        if self.stride == 1:
            return self.fr(x)
        else:
            path1 = self.path1(x)

            # pad the right and the bottom, then crop to include those pixels
            path2 = F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0.)
            path2 = path2[:, :, 1:, 1:]
            path2 = self.path2(path2)

            out = torch.cat([path1, path2], dim=1)
            out = self.bn(out)
            return out


# 定义了一个继承自nn.Module的Python类，用于表示一个ENAS模型中的神经网络层
class ENASLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes):
        super(ENASLayer, self).__init__()

        self.layer_id = layer_id  # 保存当前层的ID。
        self.in_planes = in_planes  # 保存输入通道数。
        self.out_planes = out_planes  # 保存输出通道数。

        # 定义了6种候选操作和归一化操作。
        self.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3)
        self.branch_1 = ConvBranch(in_planes, out_planes, kernel_size=3, separable=True)
        self.branch_2 = ConvBranch(in_planes, out_planes, kernel_size=5)
        self.branch_3 = ConvBranch(in_planes, out_planes, kernel_size=5, separable=True)
        self.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
        self.branch_5 = PoolBranch(in_planes, out_planes, 'max')
        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)

    # 定义了该层的前向传播过程，接收输入数据x、前面的所有层的输出prev_layers以及当前层的架构sample_arc作为输入参数。
    def forward(self, x, prev_layers, sample_arc):
        # 从sample_arc中获取当前层的类型，即决定使用哪个分支进行处理。
        layer_type = sample_arc[0]
        # 判断当前层是否为网络的第一层。
        if self.layer_id > 0:
            # 如果不是第一层，则从sample_arc中获取应该跳跃连接的层的索引
            skip_indices = sample_arc[1]
        else:
            skip_indices = []

        # 根据layer_type的值，选择合适的分支进行处理
        if layer_type == 0:
            out = self.branch_0(x)
        elif layer_type == 1:
            out = self.branch_1(x)
        elif layer_type == 2:
            out = self.branch_2(x)
        elif layer_type == 3:
            out = self.branch_3(x)
        elif layer_type == 4:
            out = self.branch_4(x)
        elif layer_type == 5:
            out = self.branch_5(x)
        else:
            raise ValueError("Unknown layer_type {}".format(layer_type))

        # 根据设置的skip_indices，对需要跳过的先前层的输出进行相加操作
        for i, skip in enumerate(skip_indices):
            if skip == 1:
                out += prev_layers[i]

        out = self.bn(out)
        return out


# 定义了一个继承自nn.Module的Python类，表示ENAS模型中的一个固定层。
class FixedLayer(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, sample_arc):
        super(FixedLayer, self).__init__()

        self.layer_id = layer_id  # 保存当前层的ID。
        self.in_planes = in_planes  # 保存输入通道数。
        self.out_planes = out_planes  # 保存输出通道数。
        self.sample_arc = sample_arc  # 保存当前层的架构。
        self.layer_type = sample_arc[0]  # 从sample_arc中获取当前层的类型。

        if self.layer_id > 0:  # 判断当前层是否为网络的第一层。
            self.skip_indices = sample_arc[1]  # 如果不是第一层，则从sample_arc中获取应该跳过的先前层的索引
        else:
            self.skip_indices = torch.zeros(1)  # 如果是第一层，则将skip_indices设置为全0的torch张量

        # 根据layer_type的值，选择合适的分支进行处理，与ENASLayer类中的代码类似
        if self.layer_type == 0:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=3)
        elif self.layer_type == 1:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=3, separable=True)
        elif self.layer_type == 2:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=5)
        elif self.layer_type == 3:
            self.branch = ConvBranch(in_planes, out_planes, kernel_size=5, separable=True)
        elif self.layer_type == 4:
            self.branch = PoolBranch(in_planes, out_planes, 'avg')
        elif self.layer_type == 5:
            self.branch = PoolBranch(in_planes, out_planes, 'max')
        else:
            raise ValueError("Unknown layer_type {}".format(self.layer_type))

        # 根据skip_indices计算输入特征图的通道数，并将其赋值给in_planes。
        in_planes = int((torch.sum(self.skip_indices).item() + 1) * in_planes)
        # 定义了一个由卷积、ReLU激活函数和批量归一化层组成的降维层
        self.dim_reduc = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_planes, track_running_stats=False))

    # 定义了该层的前向传播过程，接收输入数据x、前面的所有层的输出prev_layers以及当前层的架构sample_arc作为输入参数。
    def forward(self, x, prev_layers, sample_arc):
        out = self.branch(x)  # 使用分支处理输入。
        # 定义一个空列表，用于保存跳过的先前层的输出。
        res_layers = []
        # 遍历skip_indices列表中的每个元素以及其索引。
        for i, skip in enumerate(self.skip_indices):
            if skip == 1:
                # 如果skip的值为1，表示应该跳跃连接对应索引的前一层的输出。
                # 将前一层的输出prev_layers[i]添加到res_layers列表中。
                res_layers.append(prev_layers[i])
        # 将跳过的先前层的输出以及当前层的输出out组成一个列表。
        prev = res_layers + [out]
        # 对prev中的张量进行连接，沿着通道维度（dim=1）进行拼接。
        prev = torch.cat(prev, dim=1)
        # 通过降维层对连接后的prev进行处理。
        out = self.dim_reduc(prev)
        return out


class SeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, bias):
        super(SeparableConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
                                   padding=padding, groups=in_planes, bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L483
    '''

    def __init__(self, in_planes, out_planes, kernel_size, separable=False):
        super(ConvBranch, self).__init__()
        assert kernel_size in [3, 5], "Kernel size must be either 3 or 5"

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.separable = separable

        self.inp_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU())

        if separable:
            self.out_conv = nn.Sequential(
                SeparableConv(in_planes, out_planes, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False),
                nn.ReLU())
        else:
            padding = (kernel_size - 1) // 2
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_planes, track_running_stats=False),
                nn.ReLU())

    def forward(self, x):
        out = self.inp_conv1(x)
        out = self.out_conv(out)
        return out


class PoolBranch(nn.Module):
    '''
    https://github.com/melodyguan/enas/blob/master/src/cifar10/general_child.py#L546
    '''

    def __init__(self, in_planes, out_planes, avg_or_max):
        super(PoolBranch, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.avg_or_max = avg_or_max

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes, track_running_stats=False),
            nn.ReLU())

        if avg_or_max == 'avg':
            self.pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        elif avg_or_max == 'max':
            self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        return out


class SharedCNN(nn.Module):
    def __init__(self,
                 num_layers=12,
                 num_branches=6,
                 out_filters=24,
                 keep_prob=1.0,
                 fixed_arc=None
                 ):
        super(SharedCNN, self).__init__()

        self.num_layers = num_layers  # 网络的层数，默认为12。
        self.num_branches = num_branches  # 网络中的分支数，默认为6
        self.out_filters = out_filters  # 输出通道数，默认为24
        self.keep_prob = keep_prob  # 保持比例（用于dropout），默认为1.0（即没有dropout）
        self.fixed_arc = fixed_arc  # 一个可选的固定架构，默认为None。

        pool_distance = self.num_layers // 3  # 计算池化层之间的间隔，这里是根据网络层数除以3取整得到的。
        self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]  # 存储池化层的索引列表，其中包括每个池化层应该放置在网络中的位置

        # 定义了一个卷积层（self.stem_conv），这是网络的初始层，用于对输入数据进行卷积操作和归一化操作。
        self.stem_conv = nn.Sequential(
            nn.Conv2d(30, out_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False))

        self.layers = nn.ModuleList([])  # 存储网络层的列表
        self.pooled_layers = nn.ModuleList([])  # 存储池化层的列表

        for layer_id in range(self.num_layers):
            if self.fixed_arc is None:
                layer = ENASLayer(layer_id, self.out_filters, self.out_filters)
            else:
                layer = FixedLayer(layer_id, self.out_filters, self.out_filters, self.fixed_arc[str(layer_id)])
            self.layers.append(layer)

            if layer_id in self.pool_layers:
                for i in range(len(self.layers)):
                    if self.fixed_arc is None:
                        self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters))
                    else:
                        self.pooled_layers.append(FactorizedReduction(self.out_filters, self.out_filters * 2))
                if self.fixed_arc is not None:
                    self.out_filters *= 2

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=1. - self.keep_prob)
        self.classify = nn.Linear(self.out_filters, 2)

        # 用于初始化卷积层的权重，用于保证ReLU激活函数下不会梯度消失或爆炸
        for m in self.modules():  # 遍历模型中的所有模块。
            if isinstance(m, nn.Conv2d):  # 判断当前模块是否为nn.Conv2d类型的卷积层。
                # 对卷积层的权重进行初始化，使用了nn.init.kaiming_uniform_函数，其中：
                # m.weight表示当前卷积层的权重。
                # mode='fan_in'表示使用与输入通道数相关的fan-in模式进行初始化。
                # nonlinearity='relu'表示使用ReLU激活函数。
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    # 定义了网络的前向传播过程（forward函数）。该函数接收两个输入参数
    # x是输入数据，sample_arc是一个字典，用于定义每个层的架构
    def forward(self, x, sample_arc):
        # 输入数据经过初始卷积层进行卷积和归一化操作
        x = self.stem_conv(x)
        # 定义了两个空的列表prev_layers和pool_count，用于存储前一层的输出和池化操作的计数。
        prev_layers = []
        pool_count = 0
        # 循环遍历每个层。
        # 在每一层中，根据sample_arc字典中的指示，将输入x和前面所有层的输出prev_layers作为输入传递给网络层。
        for layer_id in range(self.num_layers):
            # 将输入数据x、前一层的输出prev_layers和当前层的架构sample_arc[str(layer_id)]
            # 传递给网络层self.layers[layer_id]，以计算当前层的输出x。
            x = self.layers[layer_id](x, prev_layers, sample_arc[str(layer_id)])
            # 将当前层的输出x添加到prev_layers列表中，以便在下一层中使用。
            prev_layers.append(x)
            # 检查当前层是否是池化层。如果是，则进行以下操作，对前面所有层的输出进行下采样操作。
            if layer_id in self.pool_layers:
                # 遍历所有前面的层以及其输出prev_layer。
                for i, prev_layer in enumerate(prev_layers):
                    # 将prev_layer经过下采样操作更新为池化层的输出，
                    # 使用self.pooled_layers[pool_count]来执行下采样操作，
                    # 并将结果保存在prev_layers的相应位置。
                    prev_layers[i] = self.pooled_layers[pool_count](prev_layer)
                    # 递增pool_count计数器，用于下一个池化层的操作。
                    pool_count += 1
                # 将最后一层的输出作为下一层的输入x，这是为了确保在下一层中使用之前所有层的下采样操作。
                x = prev_layers[-1]

        x = self.global_avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        out = self.classify(x)

        return out
