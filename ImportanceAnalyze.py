"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: ImportanceAnalyze.py
@time: 2025/5/26 11:42
"""
import numpy as np
import inspect
import json
import argparse
from GetAST import generateASt
from Java_FindTransformations import *
from C_Cpp_FindTransformations import *
from tqdm import tqdm

def create_token_byte_map(source_code: str, token_list: list, scores: np.ndarray):
    """
    创建一个包含每个token及其精确字节偏移和分数的列表。
    偏移量基于utf-8编码的源代码。

    Args:
        source_code (str): 完整的源代码字符串。
        token_list (list): CodeBERT tokenizer生成的token列表。
        scores (np.ndarray): 与token列表对应的分数数组。

    Returns:
        list: 一个字典列表，每个字典包含 'token', 'start_byte', 'end_byte', 'score'。
    """
    token_byte_map = []
    source_bytes = source_code.encode('utf-8')
    current_byte_pos = 0

    for i, token in enumerate(token_list):
        # CodeBERT的tokenizer用'Ġ'或'_'代表空格，需去除以匹配源代码
        # 同时跳过模型的特殊token
        raw_token = token.lstrip('Ġ_')
        if not raw_token or raw_token in {'<s>', '</s>', '<cls>', '<sep>', '<pad>'}:
            continue

        score = float(scores[i]) if i < len(scores) else 0.0

        try:
            # 将token编码为字节进行搜索
            token_bytes = raw_token.encode('utf-8')
            byte_start = source_bytes.find(token_bytes, current_byte_pos)

            if byte_start != -1:
                byte_end = byte_start + len(token_bytes)
                token_byte_map.append({
                    "token": raw_token,
                    "start_byte": byte_start,
                    "end_byte": byte_end,
                    "score": score
                })
                # 推进搜索位置
                current_byte_pos = byte_end
        except Exception:
            # 处理token在剥离后无法有效编码或查找的边缘情况
            pass

    return token_byte_map


def calculate_score_for_range(token_byte_map: list, pattern_start_byte: int, pattern_end_byte: int):
    """
    计算给定字节范围内所有token的分数之和。

    Args:
        token_byte_map (list): 由 create_token_byte_map 生成的列表。
        pattern_start_byte (int): 模式的起始字节。
        pattern_end_byte (int): 模式的结束字节。

    Returns:
        float: 该范围内所有token的分数总和。
    """
    total_score = 0.0
    # token_count = 0
    for token_info in token_byte_map:
        # 检查token是否完全被包含在模式的字节范围内
        if token_info['start_byte'] >= pattern_start_byte and token_info['end_byte'] <= pattern_end_byte:
            total_score += token_info['score']
            # token_count += 1

    # average_score = total_score / token_count
    # return average_score
    return total_score


def load_preprocessed_data(jsonl_file, tokens_file, scores_file):
    """加载预处理的数据文件"""
    # 加载源代码
    with open(jsonl_file, 'r') as f:
        source_codes = [json.loads(line) for line in f]

    # 加载字典格式的tokens.json
    with open(tokens_file, 'r') as f:
        tokens_by_idx = json.load(f)

    # 错误修复：加载npz格式的重要性分数
    scores_by_idx = np.load(scores_file, allow_pickle=True)

    return source_codes, tokens_by_idx, scores_by_idx


def build_token_map(source_code: str, token_list: list, scores: np.ndarray):
    # Precompute line offsets
    lines = source_code.splitlines()
    line_starts = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1  # +1 for '\n'

    # Utility to find next occurrence of token from index
    def find_token_position(src, start_idx, token):
        max_len = len(src)
        token_len = len(token)
        while start_idx < max_len:
            if src[start_idx:start_idx + token_len] == token:
                return start_idx
            start_idx += 1
        return None

    # Start constructing token_map
    token_map = [[] for _ in lines]
    src = source_code
    src_idx = 0
    score_len = len(scores)

    for j, token in enumerate(token_list):
        raw_token = token.lstrip('_')  # ignore leading underscore for matching
        if raw_token == '' or raw_token in {'<s>', '</s>'}:
            continue  # skip special or empty tokens

        score = float(scores[j]) if j < score_len else 0.0
        match_idx = find_token_position(src, src_idx, raw_token)
        if match_idx is None:
            continue  # skip if not found

        # Determine which line the match occurs in
        line_num = 0
        while line_num + 1 < len(line_starts) and match_idx >= line_starts[line_num + 1]:
            line_num += 1

        col_start = match_idx - line_starts[line_num]
        col_end = col_start + len(raw_token)
        # if token.startswith("_"):
        #     token.replace('_', '')
        token_map[line_num].append((raw_token, line_num, col_start, col_end, score))

        # Move the cursor forward
        src_idx = match_idx + len(raw_token)

    return token_map


def walk_tree(node):  # 这里的node参数是语法树的根节点
    """递归遍历所有节点，类似于ast.walk"""
    yield node
    for child in node.children:
        yield from walk_tree(child)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_codes_path", default="test1.jsonl", type=str, help="Path to test data.")
    parser.add_argument("--tokens_list_path", default="saved_models/tokens.json", type=str, help="Path to tokens file.")
    parser.add_argument("--tokens_scores_path", default="saved_models/token_grad_norms.npz", type=str, help="Path to scores file.")
    parser.add_argument("--top_n", default=10, type=int,
                        help="Number of top transformations to save for each sample.")
    parser.add_argument("--saved_filename", default="transformation_importance_scores1.json", type=str,
                        help="Name of the output JSON file.")
    parser.add_argument("--language", default="java", type=str,
                        help="Language of the source code (e.g., 'java', 'c').")
    args = parser.parse_args()

    if args.language == 'java':
        module_name = 'Java_FindTransformations'
    elif args.language == 'c':
        module_name = 'C_Cpp_FindTransformations'
    elif args.language == 'c++':
        module_name = 'C_Cpp_FindTransformations'
    else:
        raise ValueError(f"Unsupported language: {args.language}")


    # 加载数据
    source_codes, tokens_by_idx, scores_by_idx = load_preprocessed_data(
        args.source_codes_path,
        args.tokens_list_path,
        args.tokens_scores_path
    )

    # 从FindTransformations.py动态获取所有Assess函数
    assess_functions = [
        (name, func) for name, func in globals().items()
        if name.startswith('Assess') and inspect.isfunction(func) and func.__module__ == module_name
    ]

    all_samples_results = []

    for source_code_info in tqdm(source_codes, desc="Analyzing Transformations"):

        sample_idx = source_code_info.get('idx')
        if sample_idx is None:
            continue

        str_sample_idx = str(sample_idx)

        # 安全地检查数据是否存在
        if str_sample_idx not in tokens_by_idx or str_sample_idx not in scores_by_idx:
            print(f"Warning: Data for idx {sample_idx} not found. Skipping.")
            continue

        current_code = source_code_info['func']
        current_tokens = tokens_by_idx[str_sample_idx]
        current_scores = scores_by_idx[str_sample_idx]

        if len(current_tokens) >= 510:
            print(f'idx:{sample_idx} length more than 510, skip it')
            continue

        token_map = create_token_byte_map(current_code, current_tokens, current_scores)
        root_node = generateASt(current_code, args.language)

        if not root_node:
            continue

        sample_patterns = {
            "index": sample_idx,
            "target": source_code_info.get('target', -1),
            "patterns": []
        }

        for node in walk_tree(root_node):
            for name, func in assess_functions:
                transformation_info = func(node, current_code)
                if transformation_info:
                    start_byte, end_byte = transformation_info[1], transformation_info[2]
                    pattern_score = calculate_score_for_range(token_map, start_byte, end_byte)

                    sample_patterns["patterns"].append({
                        "transformation_name": name,
                        "score": pattern_score,
                        "start_point": transformation_info[3],
                        "end_point": transformation_info[4],
                        "code_snippet": current_code[start_byte:end_byte]
                    })

        if sample_patterns["patterns"]:
            sample_patterns["patterns"].sort(key=lambda x: x['score'], reverse=True)
            top_patterns = sample_patterns["patterns"][:args.top_n]
            sample_patterns["patterns"] = top_patterns
            all_samples_results.append(sample_patterns)

    # 将结果保存到JSON文件
    output_file = args.saved_filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples_results, f, ensure_ascii=False, indent=2)

    print(f"\n分析完成。成功保存 {len(all_samples_results)} 行数据到 {output_file}")


if __name__ == '__main__':
    main()
