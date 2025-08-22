"""
@author: Xiongjiawei
@contact:213223741@seu.edu.cn
@version: 1.0.0
@time: 2025/6/17 14:08
"""
import argparse
import json
import re
from tqdm import tqdm
from collections import defaultdict
from Model.GPT import GPT
from Model.Gemini import Gemini
from Model.DeepSeek import DeepSeek
from Model.Qwen import Qwen
from utils import read_jsonl, write_jsonl, load_config, get_api_key, get_model_config


def get_model(model_name, api_key=None, model_name_config=None, config=None):

    model_name_lower = model_name.lower()
    
    # 如果没有提供config，尝试加载默认配置
    if config is None:
        try:
            config = load_config()
        except:
            config = {}
    
    # 如果没有提供api_key，从config中获取
    if api_key is None:
        api_key = get_api_key(model_name, config)
    
    # 如果没有提供model_name_config，从config中获取
    if model_name_config is None:
        model_config = get_model_config(model_name, config)
        model_name_config = model_config.get('model_name', 'default')

    if model_name_lower == "gpt":
        return GPT(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "gemini":
        return Gemini(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "deepseek":
        return DeepSeek(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "qwen":
        return Qwen(api_key=api_key, model_name=model_name_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def clean_markdown_code_block(code_str):
    """
    提取 markdown 代码块中的代码，如果不存在则尝试从原字符串提取 C/C++ 代码
    """
    if not code_str or not isinstance(code_str, str):
        return ""

    code_str = code_str.strip()

    # 查找 markdown 代码块
    if "```" in code_str:
        start_idx = code_str.find("```")
        end_idx = code_str.find("```", start_idx + 3)
        if end_idx != -1:
            # 去掉语言标签（如 ```cpp 或 ```c）
            first_line_end = code_str.find("\n", start_idx)
            if first_line_end != -1 and first_line_end < end_idx:
                code_content = code_str[first_line_end + 1:end_idx]
            else:
                code_content = code_str[start_idx + 3:end_idx]
            return code_content.strip()

    # 如果没有 markdown block，但疑似 C/C++ 代码
    if "#include" in code_str or "int main" in code_str or ";" in code_str:
        return code_str

    # 否则返回空字符串，认为无效
    return ""



def extract_java_code(text):
    """
    从LLM输出中提取Java代码 - 增强的代码提取功能
    """
    if not text:
        return ""
    
    # 首先尝试markdown代码块
    cleaned = clean_markdown_code_block(text)
    if cleaned:
        return cleaned
    
    # 如果没有markdown代码块，尝试寻找Java代码模式
    # 寻找可能的Java代码开始标记
    java_patterns = [
        r'(public\s+class\s+\w+.*?)(?=\n\n|\Z)',
        r'(class\s+\w+.*?)(?=\n\n|\Z)',
        r'(public\s+\w+.*?\{.*?\})',
        r'(\w+\s+\w+\s*\(.*?\)\s*\{.*?\})'
    ]
    
    for pattern in java_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            return matches[0].strip()
    
    # 如果都没找到，返回原文本
    return text.strip()


def validate_generated_code(code):
    """
    验证生成的代码是否有效 - 新增功能
    """
    if not code or len(code.strip()) < 10:
        return False, "Code too short"
    
    # 检查是否包含基本的Java结构
    if not any(keyword in code for keyword in ['class', 'public', 'private', 'void', 'int', 'String']):
        return False, "No Java keywords found"
    
    # 检查括号匹配
    open_braces = code.count('{')
    close_braces = code.count('}')
    if abs(open_braces - close_braces) > 2:  # 允许一些小的不匹配
        return False, "Unmatched braces"
    
    return True, "Valid"


def generate_single_candidate(model, prompt, max_retries=3):
    """
    生成单个候选代码，带重试机制 - 新增功能
    """
    for attempt in range(max_retries):
        try:
            output = model.generate(prompt)
            cleaned_code = extract_java_code(output)
            
            # 验证生成的代码
            is_valid, reason = validate_generated_code(cleaned_code)
            if is_valid:
                return cleaned_code, True
            else:
                print(f"    Attempt {attempt+1} failed validation: {reason}")
                
        except Exception as e:
            print(f"    Attempt {attempt+1} failed with error: {e}")
    
    return "", False


def generate_multiple_candidates_batch(prompt, models, num_candidates=3):
    """
    使用多个模型生成多个候选代码 - 批处理版本
    """
    all_candidates = set()
    successful_generations = 0
    
    for model_name, model in models.items():
        if len(all_candidates) >= num_candidates:
            break
            
        try:
            # 每个模型生成多个候选
            for i in range(min(3, num_candidates - len(all_candidates) + 1)):
                candidate, success = generate_single_candidate(model, prompt)
                if success and candidate:
                    all_candidates.add(candidate)
                    successful_generations += 1
                    print(f"    Generated candidate {len(all_candidates)} using {model_name}")
                
                if len(all_candidates) >= num_candidates:
                    break
                    
        except Exception as e:
            print(f"    Model {model_name} failed: {e}")
    
    return list(all_candidates)[:num_candidates]


def generate_candidate_codes(prompt_file, output_file, model_names, config_path='config.json'):
    """
    主要的候选代码生成函数 - 批处理模式
    """
    try:
        config = load_config(config_path)
        print("[✓] 配置文件加载成功")
    except Exception as e:
        print(f"[✗] 配置文件加载失败: {e}")
        return

    # 初始化所有模型
    models = {}
    for model_name in model_names:
        try:
            models[model_name] = get_model(model_name, config=config)
            print(f"[✓] {model_name} 模型初始化成功")
        except Exception as e:
            print(f"[✗] {model_name} 模型初始化失败: {e}")
    
    if not models:
        print("[✗] 没有可用的模型")
        return

    # 读取prompt数据
    data = read_jsonl(prompt_file)
    results = []
    
    success_count = 0
    total_count = len(data)

    for item in tqdm(data, desc="Generating candidate codes"):
        index = item["index"]
        prompt = item["prompt"]
        
        print(f"\n[Processing] Index={index}")
        
        # 生成候选代码
        candidates = generate_multiple_candidates_batch(prompt, models, num_candidates=3)
        
        if candidates:
            success_count += 1
            print(f"[✓] 为 index={index} 生成了 {len(candidates)} 个候选")
        else:
            print(f"[✗] 为 index={index} 生成失败")
        
        results.append({
            "index": index,
            "candidate_codes": candidates,
            "num_candidates": len(candidates)
        })

    # 写入结果
    write_jsonl(output_file, results)
    
    # 打印统计信息
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n[Statistics]")
    print(f"Total samples: {total_count}")
    print(f"Successful generations: {success_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Results written to: {output_file}")


def test_model_generation(model_name, test_prompt="Write a simple Java hello world program.", config_path='config.json'):
    """
    测试模型生成功能 - 新增调试功能
    """
    try:
        config = load_config(config_path)
        model = get_model(model_name, config=config)
        
        print(f"Testing {model_name} model...")
        candidate, success = generate_single_candidate(model, test_prompt)
        
        if success:
            print(f"[✓] {model_name} test successful")
            print(f"Generated code length: {len(candidate)}")
            print(f"First 200 chars: {candidate[:200]}...")
        else:
            print(f"[✗] {model_name} test failed")
            
    except Exception as e:
        print(f"[✗] {model_name} test error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate candidate code using multiple LLMs")
    parser.add_argument('--prompt_file', type=str, help='Path to augmented_prompts.jsonl')
    parser.add_argument('--output_file', type=str, help='Path to output candidate_codes.jsonl')
    parser.add_argument('--models', nargs='+', default=["gpt", "deepseek", "claude", "qwen"], 
                       help='List of models to use')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--test', type=str, help='Test a specific model')
    args = parser.parse_args()

    if args.test:
        # 测试模式
        test_model_generation(args.test, config_path=args.config)
    elif args.prompt_file and args.output_file:
        # 正常生成模式
        generate_candidate_codes(args.prompt_file, args.output_file, args.models, args.config)
    else:
        print("Please provide either --test MODEL_NAME or both --prompt_file and --output_file")