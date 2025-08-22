"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: model.py
@time: 2025/5/13 17:40
"""
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


class CodeBERTModel(nn.Module):  # Model类继承自nn.Module，是一个用于代码漏洞检测的模型。它封装了一个编码器，用于分析代码并预测其是否包含漏洞
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeBERTModel, self).__init__()  # 调用父类的__init__()函数（必写的）
        self.encoder = encoder  # 编码器，负责将代码转换为向量表示
        self.config = config  # 编码器的配置信息
        self.tokenizer = tokenizer  # 分词器，用于处理输入文本
        self.args = args  # 参数
        self.query = 0  # 记录模型的查询次数
        # self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, labels=None, inputs_embeds=None):  # 模型的核心，定义了前向传播过程
        # input_ids：输入的代码文本token化后的ID序列、labels：输入样本的标签
        if inputs_embeds is not None:
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=input_ids.ne(1))[0]
        else:
            outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]  # 调用self.encoder处理输入，获取编码后的表示
        """
        self.encoder是一个预训练的编码器模型
        在PyTorch中，ne函数用于逐元素比较两个张量或一个张量和一个标量，判断它们是否不相等，如果不相等返回True，否则返回False 在文本处理中，1是一个特殊的填充标记，用于填充输入序列，使其长度一致
        attention_mask是与input_ids相同形状的布尔张量，用于告诉模型哪些位置是有效的输入（有效的是True），哪些是填充的（填充的是False）
        编码器的输出通常是一个元组（包含如隐藏状态、注意力权重等），其中第一个元素是最后一层的隐藏状态（即编码后的向量表示）
        """
        logits = outputs  # 直接将编码器的输出（最后一层的隐藏状态）作为logits（编码后的向量表示）
        # logits是指神经网络最后一层（通常是全连接层）的未经激活函数处理的原始输出，可以是任意实数。
        prob = torch.sigmoid(logits)  # 通过sigmoid函数将logits转换为概率值，sigmoid函数将任意实数映射到(0,1)区间，适用二分类
        # prob是模型预测的概率值，表示模型对输入代码文本中每个位置（token）是否存在漏洞的预测概率 每个元素表示对应位置token表示漏洞的概率

        if labels is not None:  # 如果提供了标签（训练或评估模式）
            labels = labels.float()  # 将labels转换为浮点型张量
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            """
            在损失计算中，使用了prob[:,0]，取每个样本序列中第一个位置token（索引0）的预测概率。这个位置通常对应特殊的[CLS]标记，在BERT类模型中，这个标记被用来表示整个序列的分类结果
            二元交叉熵损失函数公式：-(y*log(p)+(1-y)*log(1-p))  当y=1时，-log(p) 当y=0时，-log(1-p)
            1e-10是一个小常数，防止log(0)导致的数值不稳定
            """
            loss = -loss.mean()  # 对批次中所有样本的损失取平均并取负
            return loss, prob
        else:  # 如果没有提供标签（推理模式）
            return prob, logits  # 只返回概率值，不计算损失

    def get_results(self, dataset, batch_size):  # 对给定的数据集进行评估，并返回每个样本的预测概率和预测标签
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)  # 顺序采样器，按顺序遍历数据集
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()  # 将模型设置为评估模式 该模式通常在模型训练完成后，使用测试数据进行评估时使用
        logits = []  # 初始化空列表来存储logits
        labels = []  # 初始化空列表来存储labels(标签)
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")  # 将输入数据移动到 GPU
            label = batch[1].to("cuda")  # 将标签数据移动到 GPU
            with torch.no_grad():  # 关闭梯度计算，减少内存占用，提高计算速度
                lm_loss, logit = self.forward(inputs, label)
                logits.append(logit.cpu().numpy())  # 将预测的logits转换为numpy数组，并添加到列表
                labels.append(label.cpu().numpy())  # 将标签转换为numpy数组，并添加到列表中

        logits = np.concatenate(logits, 0)  #
        labels = np.concatenate(labels, 0)  #

        probs = [[1 - prob[0], prob[0]] for prob in logits]  # 计算每个样本的预测概率，prob[0]表示有漏洞的概率，1-prob[0]表示没漏洞的概率
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]  # 根据预测概率生成预测标签

        return probs, pred_labels  # 返回预测概率和预测标签
