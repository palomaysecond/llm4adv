"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: run.py
@time: 2025/5/13 17:48
"""
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import logging
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
import torch.nn as nn
logger = logging.getLogger(__name__)


class CodeBERTInputFeatures(object):  # 用于存储单个训练或测试样本的特征，这个类的设计主要是为了方便组织和访问每个样本的不同特征
    # 单个训练/测试样本的特征 (A single training/test features for a example)
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens  # 经过分词器处理后的输入代码的tokens序列
        self.input_ids = input_ids  # 将token转换为模型可以理解的数字ID序列
        self.idx=idx  # 样本的唯一标识符
        self.label=label  # 样本的标签

def codebert_convert_examples_to_features(js,tokenizer,args):  # 负责将原始的JSON格式数据转换为模型可以处理的InputFeatures对象
    #source
    # 1、处理源代码
    code=' '.join(js['func'].split())  # 去除输入代码多余的空格，只保留单词之间的单个空格  .split()+' '.join(...)
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]  # 使用tokenizer将代码文本分割成token，并对分词结果截断，[:args.block_size-2]表示从分词结果中提取前args.block_size-2个token
    # 2、添加特殊token
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]  # 在代码token序列的开头添加[CLS]，末尾添加[SEP]  是之前已经分词并截断后的代码token列表
    # 3、转换为ID
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)  # 将token序列转换为对应的ID列表  每个token在分词器的词汇表中都有一个唯一的ID
    # 4、填充，确保每个输入序列的长度一致，便于模型进行批量处理
    padding_length = args.block_size - len(source_ids)  # 计算需要填充的长度，使得当前的source_ids长度达到args.block_size  args.block_size是截断阶段提到的块大小参数，表示每个输入序列的最大长度
    source_ids+=[tokenizer.pad_token_id]*padding_length  # 使用填充token的ID进行填充，tokenizer.pad_token_id是分词器中的一个特殊token的ID，通常用于填充序列，使其达到固定长度，即将source_ids列表扩展到args.block_size的长度

    return CodeBERTInputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))


class CodeBERTTextDataset(Dataset):  # 加载和预处理代码漏洞检测数据集
    def __init__(self, tokenizer, args, file_path=None):  # 接收tokenizer、参数和文件路径作为输入
        self.examples = []  # self相当于全局变量，因为函数内部变量无法传递但self可以 构造函数里的self.examples在后面的函数里都可以用
        # 创建一个空列表self.examples来存储所有样本

        file_type = file_path.split('/')[-1].split('.')[0]  # 分割file_path取最后一个元素，按.分割后取第一个元素，即找到输入的数据集的类型，是test还是train?
        folder = '/'.join(file_path.split('/')[:-1])  # 去掉file_path最后一个'/xxx'
        cache_file_path = os.path.join(folder, 'codebert_cached_{}'.format(file_type))

        try:
            self.examples = torch.load(cache_file_path)
        except:
            with open(file_path) as f:
                for line in f:  # 打开指定文件，逐行读取并处理
                    js = json.loads(line.strip())  # 每行是一个JSON格式的字符串，使用json.loads解析为字典js
                    self.examples.append(codebert_convert_examples_to_features(js, tokenizer, args))  # 调用convert_examples_to_features函数将字典js转换为InputFeatures对象
            torch.save(self.examples, cache_file_path)

        # with open(file_path) as f:
        #     for line in f:  # 打开指定文件，逐行读取并处理
        #         js=json.loads(line.strip())  # 每行是一个JSON格式的字符串，使用json.loads解析为字典js
        #         self.examples.append(codebert_convert_examples_to_features(js, tokenizer, args))  # 调用convert_examples_to_features函数将字典js转换为InputFeatures对象
        #         # 转换后将结果存储在 self.examples 列表中

        if 'train' == file_type:  # 如果是训练文件（文件路径包含'train'） 主要目的是在训练过程中打印一些样本信息用于调试和验证
            for idx, example in enumerate(self.examples[:3]):  # 打印前3个样本的信息（包括索引、标签、输入token和输入ID）用于调试
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    # 将input_tokens中的每个token中的特殊字符\u0120（即空格）替换为_，然后将处理后的token列表打印到日志中
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    # map函数将example.input_ids中的每个ID转换为字符串，' '.join()将字符串列表用空格连接成一个字符串

    def __len__(self):  # 返回数据集中样本的数量  可以直接len(实例化的对象)返回数据集长度
        return len(self.examples)

    def __getitem__(self, i):  # 改写关于"索引"的内置方法，TextDataset[idx] 会返回该方法的return的内容
        # 根据索引i获取数据集中的第i个样本，返回的是一个元组，包含三个张量：输入ID、标签和索引
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(self.examples[i].idx)  # 将样本的input_ids、label和idx转换为PyTorch张量


def evaluate(args, model, tokenizer):
    """评估模型在测试数据集上的性能，并计算梯度归因和L2范数"""
    eval_dataset = CodeBERTTextDataset(tokenizer, args, args.test_data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # 获取嵌入层
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
    logits = []
    labels = []
    tokens_by_idx_to_save = {}
    grads_by_idx_to_save = {}
    tokens_by_idx_for_lookup = {example.idx: example.input_tokens for example in eval_dataset.examples}

    for batch in tqdm(eval_dataloader, desc="Evaluating and computing attributions"):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        input_ids.requires_grad_(False)
        label1 = label
        label_for_gather = label1.unsqueeze(1)
        batch_indices = batch[2].cpu().numpy()

        embedding = embedding_layer(input_ids)
        embedding.requires_grad_(True)
        attention_mask = input_ids.ne(1)

        # 🔹修改位置 1：自动处理 logits → 正类概率
        logits_raw, _ = model(inputs_embeds=embedding, attention_mask=attention_mask)

        if model.config.num_labels == 1:
            # 二分类 (sigmoid)
            prob_of_correct_class = torch.sigmoid(logits_raw).squeeze(-1)
        else:
            # 多分类 (softmax)
            prob_softmax = torch.softmax(logits_raw, dim=-1)
            prob_of_correct_class = torch.gather(prob_softmax, 1, label_for_gather).squeeze()

        # 🔹修改位置 2：保持 prob_diff 逻辑一致
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

        logits.append(logits_raw.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    # 保存 tokens 和梯度
    with open(os.path.join(args.output_dir, "tokens.json"), 'w', encoding='utf-8') as f:
        json.dump(tokens_by_idx_to_save, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(args.output_dir, "token_grad_norms.npz"), **grads_by_idx_to_save)

    return True



def predict_vulnerability(code, model, tokenizer, args):
    """
    对单个代码片段进行漏洞预测并提取输入嵌入层(input embeddings)的值

    Args:
        code (str): 要检测的代码片段
        model: 已加载的模型
        tokenizer: 已加载的tokenizer
        args: 参数

    Returns:
        tuple: (是否有漏洞, 漏洞概率, 输入嵌入, 嵌入信息)
    """
    # 构造输入特征
    js = {"func": code, "idx": 0, "target": 0}  # 标签无关紧要，因为我们只关心预测结果
    feature = codebert_convert_examples_to_features(js, tokenizer, args)

    # 转换为dataset
    dataset = []
    dataset.append((torch.tensor([feature.input_ids]), torch.tensor(feature.label), torch.tensor(feature.idx)))

    # 获取嵌入层
    active_path = 'encoder.roberta.embeddings.word_embeddings'
    try:
        parts = active_path.split('.')
        embedding_layer = model
        for part in parts:
            embedding_layer = getattr(embedding_layer, part)
        print(f"使用嵌入层路径: {active_path}")
    except AttributeError:
        raise ValueError(f"无法找到嵌入层路径 {active_path}，请检查模型结构")

    # 预测
    model.eval()

    inputs = torch.tensor([feature.input_ids]).to(args.device)

    # 提取输入嵌入值
    # input_embeddings = embedding_layer(inputs).detach().cpu().numpy()
    input_embeddings = embedding_layer(inputs)

    # 获取嵌入层权重信息
    embedding_weights = embedding_layer.weight.detach().cpu().numpy()

    # 计算嵌入信息
    embedding_info = {
        "active_path": active_path,
        "embedding_shape": input_embeddings.shape,
        "vocab_size": embedding_weights.shape[0],
        "embedding_dim": embedding_weights.shape[1],
        "tokens": [tokenizer.convert_ids_to_tokens(token_id) for token_id in feature.input_ids]
    }
    nums = 0
    all_nums = 0
    pad_nums = 0
    for token_id in feature.input_ids:
        all_nums += 1
        if tokenizer.convert_ids_to_tokens(token_id) != '<pad>':
            nums += 1
        else:
            pad_nums += 1
    # print(f"所有token的数量：{all_nums}")
    # print(f"非padding数量：{nums}")
    # print(f"padding数量：{pad_nums}\n")
    # print(f"嵌入向量形状: {input_embeddings.shape}")
    # print(f"词汇表大小: {embedding_weights.shape[0]}")
    # print(f"嵌入维度: {embedding_weights.shape[1]}")

    outputs = model(input_ids = inputs, inputs_embeds = input_embeddings)
    # prob1 = outputs.cpu().numpy()[0][0]
    prob = outputs[0][0]

    is_vulnerable = prob > 0.5

    prob_diff = prob - (1 - prob)
    print(type(prob_diff))
    print(type(input_embeddings))

    embedding = torch.tensor(input_embeddings, dtype=torch.float32, requires_grad=True)

    emb_grad = torch.autograd.grad(
    outputs=prob_diff,
    inputs=input_embeddings,
    retain_graph=True,
    allow_unused=True
)[0]

    token_l2 = torch.norm(emb_grad, p=2, dim=2)  # 对每个 token 的嵌入求范数

    # 归一化 只考虑非零位置的最大最小归一化
    non_zero = token_l2 != 0
    valid_grad = token_l2[non_zero]
    # 执行归一化
    normed_grad = torch.zeros_like(token_l2)
    normed_grad[non_zero] = (valid_grad - valid_grad.min()) / (valid_grad.max() - valid_grad.min() + 1e-8)

    return is_vulnerable, prob, input_embeddings, embedding_info, emb_grad, token_l2, normed_grad


def vulnerability_detect(code, model, tokenizer, args):
    """
    对单个代码片段进行漏洞检测
    Args:
        code (str): 要检测的代码片段
        model: 已加载的模型
        tokenizer: 已加载的tokenizer
        args: 参数

    Returns:
        tuple: (是否有漏洞, 漏洞概率, 输入嵌入, 嵌入信息)
    """

    # 构造输入特征
    js = {"func": code, "idx": 0, "target": 0}  # 标签无关紧要，因为我们只关心预测结果
    feature = codebert_convert_examples_to_features(js, tokenizer, args)

    # 转换为dataset
    dataset = []
    dataset.append((torch.tensor([feature.input_ids]), torch.tensor(feature.label), torch.tensor(feature.idx)))

    # 预测
    model.eval()

    inputs = torch.tensor([feature.input_ids]).to(args.device)

    prob, logit = model(input_ids=inputs)

    # logit = outputs[0][0]
    #
    # prob = torch.sigmoid(logit)

    is_vulnerable = logit > 0

    prob_diff = prob - (1 - prob)


    return is_vulnerable, logit, prob