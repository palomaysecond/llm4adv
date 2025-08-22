"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: run.py
@time: 2025/7/8 21:29
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pandas as pd
import random
import json
import re
import torch
import numpy as np
import pickle

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import tqdm
from tqdm import tqdm

from model import *
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import sys

logger = logging.getLogger(__name__)

class UniXcoderInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, input_tokens, input_ids, index, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index = index
        self.label = label


def unixcoder_convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    # 分词与截断
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    # 添加特殊标记
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    # 转换为ID，模型不认识字符串，只认识数字
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    # 填充
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return UniXcoderInputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))

class UniXcoderTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        m = re.search(r"(train|valid|test)", file_path)
        # 在 file_path 的字符串中查找是否包含 "train"、"valid" 或 "test" 这几个子字符串
        if m is None:  # 在文件路径中没有找到 "train"、"valid" 或 "test"
            partition = None
        else:
            partition = m.group(1)  # 将匹配到的字符串（"train", "valid" 或 "test"）赋值给 partition 变量

        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(unixcoder_convert_examples_to_features(js, tokenizer, args))



        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

        # self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label),
            torch.tensor(self.examples[i].index)  # 新增index
        )

def evaluate(args, model, tokenizer):
    eval_dataset = UniXcoderTextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    active_path = 'encoder.roberta.embeddings.word_embeddings'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        print(f"使用嵌入层路径: {active_path}")
    except AttributeError:
        raise ValueError(f"无法找到嵌入层路径 {active_path}，请检查模型结构")

    # 评估
    model.eval()
    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_for_lookup = {example.index: example.input_tokens for example in eval_dataset.examples}

    for batch in tqdm(eval_dataloader, desc="Evaluating and computing attributions for UniXcoder"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        batch_indices = batch[2].cpu().numpy()

        input_ids.requires_grad_(False)
        label_for_gather = label.unsqueeze(1)
        # batch_tokens = [tokens_by_idx_for_lookup[idx] for idx in batch_indices]

        embedding = embedding_layer(input_ids)
        embedding.requires_grad_(True)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        # 1. 先调用模型，获得包含两个类别概率的 prob
        #    因为我们修改了 model.py，这里会返回 (prob, logits)
        prob, _ = model(inputs_embeds=embedding, attention_mask=attention_mask)
        # print(f"prob is {prob}")
        # print(f"shape is {prob.shape}")
        # print(f"logits is {_}")

        # 2. 然后，使用这个 prob 和真实标签，挑出正确类别的概率
        prob_of_correct_class = torch.gather(prob, 1, label_for_gather).squeeze()

        # 3. 最后，根据正确类别的概率计算 prob_diff
        prob_diff = prob_of_correct_class - (1 - prob_of_correct_class)

        batch_token_grads = []
        for i in range(input_ids.size(0)):
            grad_i = torch.autograd.grad(
                outputs=prob_diff[i],
                inputs=embedding,
                retain_graph=True
            )[0][i]
            token_l2 = torch.norm(grad_i, p=2, dim=1)

            non_zero = token_l2 != 0
            valid_grad = token_l2[non_zero]

            normed_grad = torch.zeros_like(token_l2)
            if torch.numel(valid_grad) > 0 and valid_grad.max() > valid_grad.min():
                normed_grad[non_zero] = (valid_grad - valid_grad.min()) / (valid_grad.max() - valid_grad.min() + 1e-8)

            batch_token_grads.append(normed_grad.detach().cpu().numpy())

        # 使用原始idx作为键来填充字典
        for i, original_idx in enumerate(batch_indices):
            serializable_tokens = [str(t) for t in tokens_by_idx_for_lookup[original_idx]]
            tokens_by_idx_to_save[str(original_idx)] = serializable_tokens
            grads_by_idx_to_save[str(original_idx)] = batch_token_grads[i]

    # 保存为字典格式的JSON
    tokens_output_path = os.path.join(args.output_dir, "tokens.json")
    with open(tokens_output_path, 'w', encoding='utf-8') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)


    # 将梯度分数保存为 .npz 压缩包
    grads_output_path = os.path.join(args.output_dir, "token_grad_norms.npz")
    np.savez(grads_output_path, **grads_by_idx_to_save)


    return True