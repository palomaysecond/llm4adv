import torch
import os
import time
from datetime import datetime
import argparse
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from model import CodeBERTModel
from run import CodeBERTTextDataset, predict_vulnerability, evaluate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    # 基础参数设置
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")  # roberta

    # ✅ 修改位置1: 支持本地 tokenizer 路径
    parser.add_argument("--tokenizer_name", default='/root/autodl-tmp/codebert-base', type=str,
                        help="Path to pretrained tokenizer (local directory or HuggingFace repo).")

    # ✅ 修改位置2: 支持本地 model 路径
    parser.add_argument("--model_name_or_path", default='/root/autodl-tmp/codebert-base', type=str,
                        help="Path to pretrained model (local directory or HuggingFace repo).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--output_dir", default="./", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default="../dataset/Devignn/test1.jsonl", type=str,
                        help="Path to test data file.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store pretrained models.")

    # ✅ 修改位置3: 新增 checkpoint_path 参数
    parser.add_argument("--checkpoint_path", default="saved_models/checkpoint-best-acc/codebert_model.bin", type=str,
                        help="Path to fine-tuned checkpoint (.bin file)")

    args = parser.parse_args()

    # 设置设备和随机种子
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 模型加载
    config_class, model_class, tokenizer_class = {
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    }[args.model_type]

    # 从本地加载配置和分词器
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.num_labels = 1  # 表明是一个二分类任务

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=False,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # 加载预训练模型
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # 包装自定义模型
    model = CodeBERTModel(model, config, tokenizer, args)

    # ✅ 修改位置4: 使用用户指定的 checkpoint_path 加载微调权重
    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    print(f"[INFO] Loading fine-tuned weights from: {args.checkpoint_path}")

    state_dict = torch.load(args.checkpoint_path, map_location=args.device)
    # 跳过分类头
    for k in list(state_dict.keys()):
        if "classifier" in k:
            print(f"Skipping incompatible key: {k}")
            del state_dict[k]
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=args.device), strict=False)
    model.to(args.device)

    # 评估模型
    result = evaluate(args, model, tokenizer)
    print("Evaluation result:", result)


if __name__ == '__main__':
    main()