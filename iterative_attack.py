import os
import json
import random
import argparse
from tqdm import tqdm

from utils import read_jsonl, write_jsonl, get_api_key, get_model_config
from run import vulnerability_detect, CodeBertTextDataset
from generate_candidate_codes import get_model, clean_markdown_code_block
from Augmented_prompts_generator import generate_augmented_prompt

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from utils import set_seed
from model import CodeBERTModel
from UniXcoder.run import UniXcoderTextDataset
import importlib
from CodeT5.model import CodeT5Model
from CodeT5.run import CodeT5TextDataset
from UniXcoder.model import UniXcoderModel

def build_transformation_pool(language: str):
    if language.lower() == 'java':
        module_name = 'Java_FindTransformations'
    elif language.lower() in {'c', 'cpp', 'c++'}:
        module_name = 'C_Cpp_FindTransformations'
    else:
        raise ValueError(f"Unsupported language: {language}")

    module = importlib.import_module(module_name)
    transformation_funcs = {
        fn_name.replace("Assess", "").replace("_new", ""): getattr(module, fn_name)
        for fn_name in dir(module)
        if fn_name.startswith("Assess") and callable(getattr(module, fn_name))
    }
    return transformation_funcs

def CalculateFunction(outputs, outputs_original, label):
    return outputs_original[0][0] - outputs[0][0]

def generate_multiple_candidates(original_code, transformations, target_llm, num_candidates=3, language="java"):
    from Augmented_prompts_generator import generate_augmented_prompt
    candidates = []
    for i in range(num_candidates):
        try:
            prompt = generate_augmented_prompt(original_code, transformations, language=language, template_index=i)
            candidate_raw = target_llm.generate(prompt)
            candidate_clean = clean_markdown_code_block(candidate_raw)
            candidates.append(candidate_clean)
            print(f"    Generated candidate {i+1}/{num_candidates} using template {i}")
        except Exception as e:
            print(f"    Failed to generate candidate {i+1}: {e}")
            candidates.append(original_code)  # fallback
    return candidates

def attack(sample_idx, original_code, original_pred, original_logits, label,
           tokenizer, model, args, transformation_pool,
           model_name: str, target_llm, language="java", max_iteration=5, max_transformation=3):

    tmp_transformations = []
    max_score = 0
    max_transformations = []
    best_code = original_code
    queries_num = 0
    final_pred = original_pred  # === 修改位置 1: 初始化最终预测值 ===

    for iteration in range(1, max_iteration + 1):
        print(f"Iteration {iteration}")

        if len(tmp_transformations) == max_transformation:
            random_idx = random.randint(0, len(tmp_transformations) - 1)
            removed = tmp_transformations.pop(random_idx)
            print(f"Removed transformation: {removed['transformation_name']}")  

        used_names = set(t['transformation_name'] for t in tmp_transformations)
        remaining = [t for t in transformation_pool if t['transformation_name'] not in used_names]
        if not remaining:
            print("No more transformations available")
            break

        next_t = sorted(remaining, key=lambda x: -x["score"])[0]
        tmp_transformations.append(next_t)
        print(f"Added transformation: {next_t['transformation_name']} (score: {next_t['score']})")
        print(f"Current transformations: {[t['transformation_name'] for t in tmp_transformations]}")

        candidate_codes = generate_multiple_candidates(original_code, tmp_transformations, target_llm, num_candidates=3, language=language)

        attack_success = False
        success_code = None
        success_transformations = None

        for idx, candidate_code in enumerate(candidate_codes):
            print(f"  Evaluating candidate {idx+1}/{len(candidate_codes)}")
            is_vuln, logits, prob = vulnerability_detect(candidate_code, model, tokenizer, args)
            queries_num += 1
            final_pred = int(is_vuln)  # === 修改位置 2: 更新最终预测值 ===

            # === 修改位置 3: 判断预测是否与原始标签反转 ===
            # if not is_vuln and not attack_success:
            # if not final_pred and not attack_success:
            if final_pred != label:
                print(f"[\u2713] Attack success at iteration {iteration}, candidate {idx+1}")
                attack_success = True
                success_code = candidate_code
                success_transformations = tmp_transformations.copy()
                break  # 成功后退出内层循环

            score = CalculateFunction(logits, original_logits, label)
            print(f"    Candidate {idx+1} score: {score}")
            if score > max_score:
                max_score = score
                max_transformations = tmp_transformations.copy()
                best_code = candidate_code
                print(f"    New best score: {max_score}")

        if attack_success:
            break  # 成功后退出外层循环

    return success_code if attack_success else best_code, \
           success_transformations if attack_success else max_transformations, \
           attack_success, queries_num, label, final_pred  # === 修改位置 4: 返回原始和最终标签 ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="roberta", type=str)
    parser.add_argument("--model_name", default="codebert", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="../saved_models", type=str)
    parser.add_argument("--eval_data_file", default="../dataset/Devignn/test1.jsonl", type=str)
    parser.add_argument("--block_size", default=512, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--target_llm", default="deepseek", type=str)
    parser.add_argument("--language", default="java", type=str)
    parser.add_argument("--importance_score_file", default="transformation_importance_scores.json", type=str)
    parser.add_argument("--result_file", default="attack_results.jsonl", type=str)
    parser.add_argument("--finetuned_model_path", required=True, type=str,
                        help="Path to fine-tuned model weights (e.g., saved_models/codebert_model.bin)")

    args = parser.parse_args()

    if args.model_name == 'codebert':
        if not args.tokenizer_name:
            args.tokenizer_name = '../microsoft/codebert-base'
        if not args.model_name_or_path:
            args.model_name_or_path = '../microsoft/codebert-base'

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # config_class, model_class, tokenizer_class = {
    #     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    # }[args.model_type]

    # if args.model_name == 'codet5':
    #     from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
    #     config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, T5Tokenizer
    # else:
    #     config_class, model_class, tokenizer_class = {
    #         'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
    #     }[args.model_type]


    # config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir or None)
    # if args.model_name == 'codebert':
    #     config.num_labels = 1
    # elif args.model_name in {'graphcodebert', 'unixcoder'}:
    #     config.num_labels = 2
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=False, cache_dir=args.cache_dir or None)
    # args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    # model = model_class.from_pretrained(args.model_name_or_path, config=config, from_tf=bool('.ckpt' in args.model_name_or_path), cache_dir=args.cache_dir or None)
    # === 省略 imports ===
    # 【MODIFIED】调整 codet5 的 tokenizer 加载逻辑
    if args.model_name == 'codet5':
        from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer  # 【MODIFIED】
        config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer  # 【MODIFIED】
    else:
        config_class, model_class, tokenizer_class = {
            'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        }[args.model_type]


    # if args.model_name == 'codet5':
    #     from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
    #     config_class, model_class, tokenizer_class = T5Config, T5ForConditionalGeneration, RobertaTokenizer
    # elif args.model_name in {'codebert', 'graphcodebert', 'unixcoder'}:
    #     from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
    #     config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
    # else:
    #     raise ValueError(f"Unsupported model_name: {args.model_name}")

    config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir or None)
    if args.model_name == 'codebert':
        config.num_labels = 1
    elif args.model_name in {'graphcodebert', 'unixcoder'}:
        config.num_labels = 2

    # 【MODIFIED】RobertaTokenizer 不要传 do_lower_case
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        cache_dir=args.cache_dir or None
    )

    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        cache_dir=args.cache_dir or None
    )

    # === 修改位置 5: 确认所有模型都用 CodeBertTextDataset ===
    if args.model_name == 'codebert':
        model = CodeBERTModel(model, config, tokenizer, args)
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        model = CodeBERTModel(model, config, tokenizer, args)
        eval_dataset = GraphCodeBERTTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'unixcoder':
        model = UniXcoderModel(model.encoder, config, args)
        eval_dataset = UniXcoderTextDataset(tokenizer, args, file_path=args.eval_data_file)
    elif args.model_name == 'codet5':
        model = CodeT5Model(model, config, tokenizer, args)
        eval_dataset = CodeT5TextDataset(tokenizer, args, args.eval_data_file)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if not os.path.isfile(args.finetuned_model_path):
        raise FileNotFoundError(f"Fine-tuned weights not found: {args.finetuned_model_path}")

    print(f"[INFO] Loading fine-tuned model from: {args.finetuned_model_path}")
    model.load_state_dict(torch.load(args.finetuned_model_path, map_location=args.device), strict=False)
    model.to(args.device)

    with open(args.eval_data_file, 'r', encoding='utf-8') as f:
        idx_to_code = {json.loads(line)['idx']: json.loads(line)['func'] for line in f}

    with open(args.importance_score_file, 'r', encoding='utf-8') as f:
        all_transformation_scores = json.load(f)
    idx_to_transformations = {item['index']: item['patterns'] for item in all_transformation_scores}
    transformation_funcs = build_transformation_pool(args.language)

    target_llm = get_model(args.target_llm)
    all_results = []
    num_code = 0
    num_code_success = 0
    queries_list = []

    for batch in tqdm(eval_dataloader, desc="Attacking the {}".format(args.model_name)):
        input_ids = batch[0].to(args.device)
        label = batch[1].to(args.device)
        index = batch[2].to(args.device)

        with torch.no_grad():
            outputs, _ = model(input_ids=input_ids)
            preds = (outputs[:, 0] > 0.5).int()

        correct_mask = (preds == label)
        if not correct_mask.any():
            continue
        correct_indices = index[correct_mask]

        for idx in correct_indices:
            sample_idx = idx.item()
            if sample_idx not in idx_to_transformations:
                print(f"Skipping idx {sample_idx}: no transformations found")
                continue

            num_code += 1
            is_vulnerable, logit, prob = vulnerability_detect(idx_to_code[sample_idx], model, tokenizer, args)

            transformation_pool = idx_to_transformations.get(sample_idx, [])
            for t in transformation_pool:
                t_name_clean = t['transformation_name'].replace("Assess", "").replace("_new", "")
                t['transformation_name'] = t_name_clean
                t['function'] = transformation_funcs.get(t_name_clean)

            label_scalar = label[correct_mask][(correct_indices == idx).nonzero(as_tuple=True)[0][0]].item()

            newcode, transformations, attackState, queries, original_label, final_pred = attack(
                sample_idx=sample_idx,
                original_code=idx_to_code[sample_idx],
                original_pred=is_vulnerable,
                original_logits=logit,
                tokenizer=tokenizer,
                model=model,
                args=args,
                transformation_pool=transformation_pool,
                model_name=args.model_name,
                target_llm=target_llm,
                label=label_scalar,
                language=args.language
            )

            result_data = {
                "idx": sample_idx,
                "status": "success" if attackState else "failed",
                "target_llm": args.target_llm,
                "language": args.language,
                "original_label": original_label,  # === 修改位置 6: 记录原始标签
                "final_prediction": final_pred     # === 修改位置 6: 记录最终预测
            }
            if attackState:
                num_code_success += 1
                queries_list.append(queries)
                result_data["new_code"] = newcode
                result_data["transformations"] = [t['transformation_name'] for t in transformations]
            else:
                queries_list.append(queries)

            all_results.append(result_data)

    attack_success_rate = round(num_code_success / num_code, 4) if num_code > 0 else 0
    if len(queries_list) > 0:
        total_queries = sum(queries_list)
        average_model_queries = round(total_queries / len(queries_list), 2)

    statistics = {
        "attack_success_rate": attack_success_rate,
        "average_model_queries": average_model_queries,
        "total_samples": num_code,
        "successful_attacks": num_code_success
    }

    with open(args.result_file, "w", encoding="utf-8") as result_file:
        result_file.write(json.dumps(statistics, ensure_ascii=False) + '\n')
        for result_item in all_results:
            result_file.write(json.dumps(result_item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
