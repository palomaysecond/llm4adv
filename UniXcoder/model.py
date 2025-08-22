"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: model.py
@time: 2025/7/8 20:33
"""
import torch
import torch.nn as nn
import torch
import itertools
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import torch.nn.functional as F

class UniXcoderModel(nn.Module):
    def __init__(self, encoder, config, args):
        super(UniXcoderModel, self).__init__()
        self.encoder = encoder  # 预训练的编码器模型
        self.config = config  # 模型的配置信息
        self.args = args  # 运行时的各种参数
        self.query = 0  # 一个计数器，用于记录模型处理的样本数量

        # self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs_embeds=None, input_ids=None, labels=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(1)
        if inputs_embeds is not None:
            logits = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0]
        else:
            logits = self.encoder(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=input_ids.ne(1))[0]
        # prob = self.softmax(logits)
        prob = F.softmax(logits, dim=-1)
        input_size = input_ids.size() if input_ids is not None else inputs_embeds.size()
        assert prob.size(0) == input_size[0], (prob.size(), input_size)
        # 确保模型输出的样本数量与输入的样本数量完全一致
        assert prob.size(1) == 2, prob.size()
        # 确保模型的输出维度是二分类结果
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, prob
        else:
            return prob, logits
        # prob 是模型对每个输入样本属于各个类别的预测概率；由原始的、未经处理的模型输出 logits 经过 nn.Softmax 函数计算得来的。
        # prob 是一个 PyTorch 张量（Tensor），它的形状是 (batch_size, num_labels)，batch_size: 指的是你一次性输入给模型的样本数量 um_labels: 指的是分类任务的类别数量
        """
        prob:tensor([[0.6923, 0.3077],
        [0.5315, 0.4685],
        [0.6317, 0.3683],
        [0.7081, 0.2919]], device='cuda:0', grad_fn=<SoftmaxBackward0>)
        
        logits: tensor([[ 0.4522, -0.3586],
        [-0.1191, -0.2452],
        [ 0.4804, -0.0590],
        [ 0.5116, -0.3748]], device='cuda:0', grad_fn=<AddmmBackward0>)
        """

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(input_ids=inputs, labels=label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = [[prob[0], 1 - prob[0]] for prob in logits]
        pred_labels = [0 if label else 1 for label in logits[:, 0] > 0.5]
        return probs, pred_labels