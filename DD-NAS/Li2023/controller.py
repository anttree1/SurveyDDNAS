import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


class Controller(nn.Module):
    def __init__(self,
                 num_layers=12,  # 搜索到的网络的层数
                 num_branches=5,  # 候选操作数
                 lstm_size=32,  # LSTM输入维度
                 lstm_num_layers=2,  # LSTM层数
                 tanh_constant=1.5,
                 temperature=None,
                 skip_target=0.4,
                 skip_weight=0.8):
        super(Controller, self).__init__()

        self.num_layers = num_layers
        self.num_branches = num_branches

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._create_params()

    def _create_params(self):
        self.w_lstm = nn.LSTM(input_size=self.lstm_size,
                              hidden_size=self.lstm_size,
                              num_layers=self.lstm_num_layers)

        self.g_emb = nn.Embedding(1, self.lstm_size)  # Learn the starting input

        self.w_emb = nn.Embedding(self.num_branches, self.lstm_size)
        self.w_soft = nn.Linear(self.lstm_size, self.num_branches, bias=False)
        self.w_soft3 = nn.Linear(self.lstm_size, 3, bias=False)
        self.w_soft4 = nn.Linear(self.lstm_size, 4, bias=False)
        self.w_soft5 = nn.Linear(self.lstm_size, 5, bias=False)
        self.w_soft6 = nn.Linear(self.lstm_size, 6, bias=False)

        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)

    def forward(self):
        h0 = None  # setting h0 to None will initialize LSTM state with 0s

        anchors = []
        anchors_w_1 = []

        arc_seq = {}
        entropys = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        # 使用预训练的权重初始化输入
        inputs = self.g_emb.weight
        skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_target]).cuda()

        # 在循环中生成神经网络的各个层
        for layer_id in range(self.num_layers):
            # 在输入的第一个维度上增加一个额外的维度，以匹配 LSTM 的输入要求
            inputs = inputs.unsqueeze(0)
            # 通过 LSTM 接收输入和之前的隐藏状态，然后生成输出和新的隐藏状态
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)
            h0 = hn

            # 通过softmax函数对输出进行处理，生成概率分布
            logit = self.w_soft(output)
            # tanh_constant 是一个常数，其作用是对模型中的某些部分进行缩放调整
            if self.tanh_constant is not None:
                logit = self.tanh_constant * torch.tanh(logit)

            # 通过这些概率分布抽样一个动作
            branch_id_dist = Categorical(logits=logit)
            branch_id = branch_id_dist.sample()

            arc_seq[str(layer_id)] = branch_id.tolist()

            log_prob = branch_id_dist.log_prob(branch_id)
            log_probs.append(log_prob.view(-1))
            entropy = branch_id_dist.entropy()
            entropys.append(entropy.view(-1))

            inputs = self.w_emb(branch_id)
            inputs = inputs.unsqueeze(0)

            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)

            if layer_id > 0:
                query = torch.cat(anchors_w_1, dim=0)
                query = torch.tanh(query + self.w_attn_2(output))
                query = self.v_attn(query)
                logit = torch.cat([-query, query], dim=1)
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                skip_dist = Categorical(logits=logit)
                skip = skip_dist.sample()
                skip = skip.view(layer_id)

                arc_seq[str(layer_id)].append(torch.sum(skip))

                skip_prob = torch.sigmoid(logit)
                kl = skip_prob * torch.log(skip_prob / skip_targets)
                kl = torch.sum(kl)
                skip_penaltys.append(kl)

                log_prob = skip_dist.log_prob(skip)
                log_prob = torch.sum(log_prob)
                log_probs.append(log_prob.view(-1))

                entropy = skip_dist.entropy()
                entropy = torch.sum(entropy)
                entropys.append(entropy.view(-1))

                skip = skip.type(torch.float)
                skip = skip.view(1, layer_id)
                skip_count.append(torch.sum(skip))
                inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
                inputs /= (1.0 + torch.sum(skip))

            else:
                inputs = self.g_emb.weight
                inp = torch.tensor(0).cuda()
                arc_seq[str(layer_id)].append(inp)

            # 第二轮
            # 在输入的第一个维度上增加一个额外的维度，以匹配 LSTM 的输入要求
            inputs = inputs.unsqueeze(0)
            # 通过 LSTM 接收输入和之前的隐藏状态，然后生成输出和新的隐藏状态
            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)
            h0 = hn

            # 通过softmax函数对输出进行处理，生成概率分布
            logit = self.w_soft(output)
            # tanh_constant 是一个常数，其作用是对模型中的某些部分进行缩放调整
            if self.tanh_constant is not None:
                logit = self.tanh_constant * torch.tanh(logit)

            # 通过这些概率分布抽样一个动作
            branch_id_dist = Categorical(logits=logit)
            branch_id = branch_id_dist.sample()

            arc_seq[str(layer_id)].append(branch_id)

            log_prob = branch_id_dist.log_prob(branch_id)
            log_probs.append(log_prob.view(-1))
            entropy = branch_id_dist.entropy()
            entropys.append(entropy.view(-1))

            inputs = self.w_emb(branch_id)
            inputs = inputs.unsqueeze(0)

            output, hn = self.w_lstm(inputs, h0)
            output = output.squeeze(0)

            if layer_id > 0:
                query = torch.cat(anchors_w_1, dim=0)
                query = torch.tanh(query + self.w_attn_2(output))
                query = self.v_attn(query)
                logit = torch.cat([-query, query], dim=1)
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * torch.tanh(logit)

                skip_dist = Categorical(logits=logit)
                skip = skip_dist.sample()
                skip = skip.view(layer_id)

                arc_seq[str(layer_id)].append(torch.add(torch.sum(skip), 1))

                skip_prob = torch.sigmoid(logit)
                kl = skip_prob * torch.log(skip_prob / skip_targets)
                kl = torch.sum(kl)
                skip_penaltys.append(kl)

                log_prob = skip_dist.log_prob(skip)
                log_prob = torch.sum(log_prob)
                log_probs.append(log_prob.view(-1))

                entropy = skip_dist.entropy()
                entropy = torch.sum(entropy)
                entropys.append(entropy.view(-1))

                skip = skip.type(torch.float)
                skip = skip.view(1, layer_id)
                skip_count.append(torch.sum(skip))
                inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
                inputs /= (1.0 + torch.sum(skip))

            else:
                inputs = self.g_emb.weight
                inp = torch.tensor(1).cuda()
                arc_seq[str(layer_id)].append(inp)

            anchors.append(output)
            anchors_w_1.append(self.w_attn_1(output))

        self.sample_arc = arc_seq

        entropys = torch.cat(entropys)
        self.sample_entropy = torch.sum(entropys)

        log_probs = torch.cat(log_probs)
        self.sample_log_prob = torch.sum(log_probs)

        skip_count = torch.stack(skip_count)
        self.skip_count = torch.sum(skip_count)

        skip_penaltys = torch.stack(skip_penaltys)
        self.skip_penaltys = torch.mean(skip_penaltys)
