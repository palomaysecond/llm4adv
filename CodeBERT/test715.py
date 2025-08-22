import sys
import os
import json
import random
import argparse
from tqdm import tqdm
from run import vulnerability_detect, GraphCodeBERTTextDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from utils import set_seed
from CodeBERT.run import CodeBERTTextDataset
# from UniXcoder.run import UniXcoderTextDataset
# from GraphCodeBERT.model import GraphCodeBERTModel
from model import CodeBERTModel
# from CodeBERT.model import CodeBERTModel
# from UniXcoder.model import UniXcoderModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name", default="codebert", type=str)
    parser.add_argument("--output_dir", default="../saved_models", type=str)
    parser.add_argument("--eval_data_file", default="../dataset/Devignn/test1.jsonl", type=str)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--target_llm", default="deepseek", type=str)
    parser.add_argument("--language", default="java", type=str)
    parser.add_argument("--importance_score_file", default="transformation_importance_scores.json", type=str)
    parser.add_argument("--result_file", default="attack_results.jsonl", type=str)
    parser.add_argument("--finetuned_model_path", required=True, type=str,
                        help="Path to fine-tuned model weights or model directory (e.g., saved_models/codebert_model.bin or saved_models/codebert/)")

    args = parser.parse_args()

    # === 修改位置 1: 优化模型路径设置 ===
    if args.model_name == 'codebert':
        args.tokenizer_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.block_size = 512
    elif args.model_name == 'graphcodebert':
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
        args.block_size = 512
    elif args.model_name == 'unixcoder':
        args.tokenizer_name = 'microsoft/unixcoder-base'
        args.model_name_or_path = 'microsoft/unixcoder-base'
        args.block_size = 512

    # === 修改位置 2: 检查是否使用本地模型目录 ===
    if os.path.isdir(args.finetuned_model_path):
        print(f"[INFO] Using local model directory: {args.finetuned_model_path}")
        args.model_name_or_path = args.finetuned_model_path
        args.tokenizer_name = args.finetuned_model_path

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }[args.model_type]

    print(f"[INFO] Loading config from: {args.model_name_or_path}")
    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir or None)
    if args.model_name == 'codebert':
        config.num_labels = 1
    elif args.model_name in {'graphcodebert', 'unixcoder'}:
        config.num_labels = 2

    print(f"[INFO] Loading tokenizer from: {args.tokenizer_name}")
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=False, cache_dir=args.cache_dir or None)

    print(f"[INFO] Loading base model from: {args.model_name_or_path}")
    model = model_class.from_pretrained(args.model_name_or_path, config=config, from_tf=bool('.ckpt' in args.model_name_or_path), cache_dir=args.cache_dir or None)

    # === 修改位置 3: 如果 finetuned_model_path 是权重文件则加载 ===
    if os.path.isfile(args.finetuned_model_path) and args.finetuned_model_path.endswith(".bin"):
        print(f"[INFO] Loading fine-tuned weights from: {args.finetuned_model_path}")
        model.load_state_dict(torch.load(args.finetuned_model_path, map_location=args.device), strict=False)

    # === 修改位置 4: 初始化特定模型 ===
    if args.model_name == 'codebert':
        model = CodeBERTModel(model, config, args)
        eval_dataset = CodeBERTTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERTModel(model, config, args)
        eval_dataset = GraphCodeBERTTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'unixcoder':
        model = UniXcoderModel(model, config, args)
        eval_dataset = UniXcoderTextDataset(tokenizer, args, args.eval_data_file)

    model.to(args.device)

    # === 修改位置 5: 简化输出验证逻辑 ===
    # index_list = []
    # label_list = []
    # ori_list = []
    # for index, example in enumerate(eval_dataset):
    #     index_list.append(example[4].item())
    #     label_list.append(example[3].item())
    #     orig_prob, orig_pred, _ = model.get_results([example], args.eval_batch_size)
    #     print(f"orig_prob: {orig_prob}, orig_pred: {orig_pred}, {_}")
    #     orig_pred = orig_pred[0]
    #     ori_list.append(orig_pred)
    #     print(f"Sample idx: {index_list[-1]}, label: {label_list[-1]}, original_pred: {ori_list[-1]}")
    #     break  # Remove this break if you want to process all samples

    index_list = []
    label_list = []
    ori_list = []
    index_num = 0
    for index, example in enumerate(eval_dataset):
        # print(example[0])
        # print(example[1].item())
        # print(example[2])
        index_list.append(example[4].item())
        label_list.append(example[3].item())
        orig_prob, orig_pred , _ = model.get_results([example], args.eval_batch_size)
        orig_prob = orig_prob[0]
        print(f"0: {_}")
        print(f"1: {orig_pred}")
        orig_pred = orig_pred[0]
        print(f"2: {orig_pred}")
        ori_list.append(orig_pred)
        # break
    for i in range(len(index_list)):
        if label_list[i] == 0 and ori_list[i] == 0:
            index_num += 1
        print(f"index: {index_list[i]}, label: {label_list[i]}, orig_pred: {ori_list[i]}")
    print(index_num)

if __name__ == "__main__":
    main()
