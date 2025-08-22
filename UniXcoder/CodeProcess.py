"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: CodeProcess.py
@time: 2025/7/9 14:47
"""

import torch
import os
import argparse
from transformers import (
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
)
from model import UniXcoderModel
from run import evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()

    # 模型架构和路径
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model architecture to be used (roberta for CodeBERT).")
    parser.add_argument("--model_name", default="unixcoder", type=str,
                        help="Name of the model (e.g., unixcoder, codebert).")
    parser.add_argument("--base_model_path", required=True, type=str,
                        help="Path to locally downloaded base model directory (e.g., ../microsoft/unixcoder-base).")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Optional: path to tokenizer directory. Defaults to base_model_path if not set.")
    parser.add_argument("--finetuned_model_path", required=True, type=str,
                        help="Path to fine-tuned model weights (e.g., saved_models/unixcoder_model.bin).")

    # 输出目录
    parser.add_argument("--output_dir", default="./", type=str,
                        help="Directory where predictions and outputs will be written. "
                             "Defaults to fine-tuned model directory if not set.")

    # 其他参数
    parser.add_argument("--test_data_file", default="../dataset/Devignn/test1.jsonl", type=str,
                        help="Path to test data file.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--language", type=str, default='c', help="Source code language (java or c/cpp).")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store pretrained models cache.")
    parser.add_argument("--config_name", type=str, default="",
                    help="Optional config name or path. Defaults to base_model_path if not set.")

    args = parser.parse_args()

    # 如果用户没指定 --output_dir，则默认用微调权重所在目录
    if not args.output_dir or args.output_dir == "./":
        args.output_dir = os.path.dirname(args.finetuned_model_path)
    print(f"[INFO] Output directory set to: {args.output_dir}")

    # 设置 block_size 和设备
    args.block_size = 512
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 加载模型配置、分词器和基础权重
    print(f"[INFO] Loading base model from: {args.base_model_path}")
    config_class, model_class, tokenizer_class = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.base_model_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 2  # 二分类任务

    tokenizer_dir = args.tokenizer_path if args.tokenizer_path else args.base_model_path
    print(f"[INFO] Loading tokenizer from: {tokenizer_dir}")
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_dir,
        do_lower_case=False,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    args.block_size = min(args.block_size, tokenizer.model_max_length)

    base_model = model_class.from_pretrained(
        args.base_model_path,
        config=config,
        from_tf=bool('.ckpt' in args.base_model_path),
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # 包装为 UniXcoderModel
    model = UniXcoderModel(base_model, config, args)

    # 加载微调模型权重
    if not os.path.isfile(args.finetuned_model_path):
        raise FileNotFoundError(f"Fine-tuned model weights not found at: {args.finetuned_model_path}")

    print(f"[INFO] Loading fine-tuned weights from: {args.finetuned_model_path}")
    model.load_state_dict(torch.load(args.finetuned_model_path, map_location=args.device), strict=False)

    # 将模型移动到设备
    model.to(args.device)

    # 评估
    result = evaluate(args, model, tokenizer)
    print("Evaluation result:", result)


if __name__ == '__main__':
    main()
