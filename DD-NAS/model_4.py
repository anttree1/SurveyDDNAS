from operations import *


class Layer(nn.Module):
    def __init__(self, out_planes, drop_size):
        super(Layer, self).__init__()

        self.out_planes = out_planes
        self.drop_size = drop_size
        self.op1 = OPS['avg3'](out_planes, stride=1, affine=False)
        self.op2 = OPS['max3'](out_planes, stride=1, affine=False)
        self.op3 = OPS['sep3'](out_planes, stride=1, affine=False)
        self.op4 = OPS['sep5'](out_planes, stride=1, affine=False)
        self.op5 = OPS['skip'](out_planes, stride=1, affine=False)
        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=False)
        self.drop_out = nn.Dropout(p=self.drop_size)

    def forward(self, prev_layers, sample_arc):
        layer_type1 = sample_arc[0]
        layer_type2 = sample_arc[2].item()
        x1 = prev_layers[sample_arc[1].item()]
        x2 = prev_layers[sample_arc[3].item()]

        if layer_type1 == 0:
            out1 = self.op1(x1)
        elif layer_type1 == 1:
            out1 = self.op2(x1)
        elif layer_type1 == 2:
            out1 = self.op3(x1)
        elif layer_type1 == 3:
            out1 = self.op4(x1)
        elif layer_type1 == 4:
            out1 = self.op5(x1)

        if layer_type2 == 0:
            out2 = self.op1(x2)
        elif layer_type2 == 1:
            out2 = self.op2(x2)
        elif layer_type2 == 2:
            out2 = self.op3(x2)
        elif layer_type2 == 3:
            out2 = self.op4(x2)
        elif layer_type2 == 4:
            out2 = self.op5(x2)
        out = out1 + out2
        out = self.drop_out(out)
        out = self.bn(out)
        return out


class Cell(nn.Module):
    def __init__(self,
                 num_layers=5,
                 out_filters=30,
                 drop_size=0.2,
                 in_size=30,
                 reduction=False
                 ):
        super(Cell, self).__init__()

        self.num_layers = num_layers
        self.out_filters = out_filters
        self.drop_size = drop_size
        self.reduction = reduction
        self.layers = nn.ModuleList([])
        if self.reduction:
            self.conv = FactorizedReduce(in_size, out_filters)
        else:
            self.conv = ReLUConvBN(in_size, out_filters, 1, 1, 0)

        for layer_id in range(self.num_layers):
            layer = Layer(self.out_filters,self.drop_size)
            self.layers.append(layer)

        # 初始化卷积层的权重，用于保证ReLU激活函数下不会梯度消失或爆炸
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x1, x2, sample_arc):
        prev_layers = []
        end_layers = []
        prev_layers.append(x1)
        prev_layers.append(x2)
        for layer_id in range(self.num_layers):
            x = self.layers[layer_id](prev_layers, sample_arc[str(layer_id)])
            prev_layers.append(x)
            end_layers.append(sample_arc[str(layer_id)][1].item())
            end_layers.append(sample_arc[str(layer_id)][3].item())
            end_layers = list(set(end_layers))

        new_list = [el for i, el in enumerate(prev_layers) if i not in end_layers]
        stacked_tensor = torch.stack(new_list, dim=0)
        out = torch.sum(stacked_tensor, dim=0)
        out = self.conv(out)
        return out


class SharedCNN(nn.Module):

    def __init__(self, drop_size, inp_size, num_cell=7, out_filters=30):
        super(SharedCNN, self).__init__()
        self.num_cell = num_cell
        self.drop_size = drop_size
        self.cells = nn.ModuleList()
        self.inp_size = inp_size
        self.out_filters = out_filters
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.drop_size)
        self.classify = nn.Linear(self.out_filters, 2)
        self.softmax = nn.Softmax(dim=1)

        self.stem_conv = nn.Sequential(
            nn.Conv2d(self.inp_size, out_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_filters, track_running_stats=False))

        for i in range(self.num_cell):
            cell = Cell(drop_size=self.drop_size)
            self.cells += [cell]

    def forward(self, input, sample_arc):
        input = self.stem_conv(input)
        s0 = s1 = input
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, sample_arc)
        out = self.global_avg_pool(s1)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        # logits = self.softmax(out)
        logits = out
        return logits
