"""
@author: Xiongjiawei
@contact:213223741@seu.edu.cn
@version: 1.1.1 (topk supported + format error handling)
@time: 2025/08/16
"""
import json
from collections import defaultdict
import random
import os

# Transformation rule descriptions for Java and C/C++
RULE_DESCRIPTIONS = {
    "java": {
        "ForToWhileConversion": "Replace 'for-loop' with equivalent 'while-loop'.",
        "WhileToForRefactoring": "Replace 'while-loop' with equivalent 'for-loop'.",
        "DoWhileToWhileConversion": "Replace 'do-while-loop' with equivalent 'while-loop'.",
        "InlineLoopDeclaration": "Move loop variable declaration into 'for' header.",
        "ExtractLoopDeclaration": "Move loop variable declaration out of 'for' header.",
        "IfElseBranchSwap": "Swap 'if' and 'else' blocks using negated conditions.",
        "ElseIfToNestedIf": "Convert 'else if' into 'else { if }' structure.",
        "NestedIfToElseIf": "Merge 'else { if }' into 'else if' structure.",
        "WrapStatementInBlock": "Add curly braces '{}' to single-statement blocks.",
        "UnwrapRedundantBlock": "Remove redundant '{}' from single-statement blocks.",
        "SplitCompoundCondition": "Split compound conditions into nested if-statements.",
        "AddRedundantStatement": "Insert no-op code like 'x = x;' to maintain semantics.",
        "WrapWithConstantCondition": "Wrap statements in always-true conditional blocks.",
        "ExtractSubexpression": "Extract part of complex expressions into temporary variables.",
        "SwitchToIfElse": "Convert 'switch' statements to equivalent 'if-else' chains.",
        "ReturnViaTempVariable": "Assign return value to a temporary variable before returning.",
        "NegateWithReversedOperator": "Replace comparisons with negated, reversed operators.",
        "ExpandCompoundAssign": "Expand compound assignments into full expressions (e.g., 'x += 1' → 'x = x + 1').",
        "ExpandUnaryOP": "Expand unary increment/decrement operators into explicit assignments.",
        "PromoteIntToLong": "Change 'int' declarations to 'long' where semantics allow.",
        "PromoteFloatToDouble": "Change 'float' declarations to 'double'.",
        "AddUnusedParameter": "Add unused parameters to method signatures.",
        "RefactorOutputAPI": "Split output API calls into separate stream references and method calls.",
        "RenameVariable": "Rename variables consistently without affecting program logic.",
        "RenameClassAndMethod": "Rename class and method identifiers while preserving behavior."
    },
    "cpp": {
        "ForToWhileConversion": "Replace 'for-loop' with equivalent 'while-loop'.",
        "WhileToForRefactoring": "Replace 'while-loop' with equivalent 'for-loop'.",
        "DoWhileToWhileConversion": "Replace 'do-while-loop' with equivalent 'while-loop'.",
        "InlineLoopDeclaration": "Move loop variable declaration into 'for' header.",
        "ExtractLoopDeclaration": "Move loop variable declaration out of 'for' header.",
        "IfElseBranchSwap": "Swap 'if' and 'else' blocks using negated conditions.",
        "ElseIfToNestedIf": "Convert 'else if' into 'else { if }' structure.",
        "NestedIfToElseIf": "Merge 'else { if }' into 'else if' structure.",
        "WrapStatementInBlock": "Add curly braces '{}' to single-statement blocks.",
        "UnwrapRedundantBlock": "Remove redundant '{}' from single-statement blocks.",
        "SplitCompoundCondition": "Split compound conditions into nested if-statements.",
        "AddRedundantStatement": "Insert no-op code like 'x = x;' to maintain semantics.",
        "WrapWithConstantCondition": "Wrap statements in always-true conditional blocks.",
        "ExtractSubexpression": "Extract part of complex expressions into temporary variables.",
        "SwitchToIfElse": "Convert 'switch' statements to equivalent 'if-else' chains.",
        "ReturnViaTempVariable": "Assign return value to a temporary variable before returning.",
        "NegateWithReversedOperator": "Replace comparisons with negated, reversed operators.",
        "ExpandCompoundAssign": "Expand compound assignments into full expressions (e.g., 'x += 1' → 'x = x + 1').",
        "ExpandUnaryOP": "Expand unary increment/decrement operators into explicit assignments.",
        "PromoteIntToLong": "Change 'int' declarations to 'long' where semantics allow.",
        "PromoteFloatToDouble": "Change 'float' declarations to 'double'.",
        "AddUnusedParameter": "Add unused parameters to function signatures.",
        "RefactorOutputAPI": "Split output API calls into separate stream references and method calls.",
        "RenameVariable": "Rename variables consistently without affecting program logic.",
        "RenameClassAndMethod": "Rename class and function identifiers while preserving behavior."
    }
}

PROMPT_TEMPLATES = {
    "java": [
        # Template 0 - Original
        '''You are a powerful AI code transformer specializing in generating adversarial variants of source code that are indistinguishable in behavior but significantly altered in structure. Your mission is to help bypass static vulnerability detectors by rewriting code using multiple semantics-preserving transformations.

Your refactored code must apply all given transformations accurately and thoroughly, modifying every occurrence and minimizing superficial or trivial changes. The goal is to reshape the code structure while strictly preserving functionality and ensuring it compiles without errors.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 1
        '''You are an expert AI in code restructuring, specializing in generating behavior-preserving but structurally divergent variants of source code. Your task is to apply a set of semantic-preserving transformations to the input code to help evade static vulnerability detection tools.

You must accurately and thoroughly perform all specified transformations, changing every relevant structure while preserving the exact behavior and ensuring the resulting code compiles correctly.

Focus on deep structural rewrites rather than superficial edits. Your transformed code should be as different as possible in form while remaining functionally identical to the original.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 2
        '''You are a high-performance adversarial code generator. Your job is to apply a series of semantics-preserving transformations to the given source code, producing a variant that maintains identical functionality but appears significantly different to static analysis tools.

Every transformation listed must be fully implemented wherever applicable. Avoid trivial rewrites — aim for maximal structural change without introducing errors or altering program behavior.

Your output must be compilable, behaviorally equivalent, and structurally altered to evade detection by conventional vulnerability scanners.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
'''
    ],
    "cpp": [
        # Template 0 - Original
        '''You are a powerful AI code transformer specializing in generating adversarial variants of source code that are indistinguishable in behavior but significantly altered in structure. Your mission is to help bypass static vulnerability detectors by rewriting code using multiple semantics-preserving transformations.

Your refactored code must apply all given transformations accurately and thoroughly, modifying every occurrence and minimizing superficial or trivial changes. The goal is to reshape the code structure while strictly preserving functionality and ensuring it compiles without errors.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 1
        '''You are an expert AI in code restructuring, specializing in generating behavior-preserving but structurally divergent variants of source code. Your task is to apply a set of semantic-preserving transformations to the input code to help evade static vulnerability detection tools.

You must accurately and thoroughly perform all specified transformations, changing every relevant structure while preserving the exact behavior and ensuring the resulting code compiles correctly.

Focus on deep structural rewrites rather than superficial edits. Your transformed code should be as different as possible in form while remaining functionally identical to the original.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 2
        '''You are a high-performance adversarial code generator. Your job is to apply a series of semantics-preserving transformations to the given source code, producing a variant that maintains identical functionality but appears significantly different to static analysis tools.

Every transformation listed must be fully implemented wherever applicable. Avoid trivial rewrites — aim for maximal structural change without introducing errors or altering program behavior.

Your output must be compilable, behaviorally equivalent, and structurally altered to evade detection by conventional vulnerability scanners.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
'''
    ]
}


def normalize_language(language: str) -> str:
    language = language.lower()
    if language in {"c", "cpp", "c++"}:
        return "cpp"
    if language == "java":
        return "java"
    raise ValueError(f"Unsupported language: {language}")


def format_transformation_rule(rule_name, loc_desc, language):
    lang_key = normalize_language(language)
    description = RULE_DESCRIPTIONS[lang_key].get(rule_name, "No description provided.")
    return f"- {rule_name} at {loc_desc}: {description}"


def generate_augmented_prompt(code_str, transformations, language="java", template_index=0):
    lang_key = normalize_language(language)
    transformation_lines = [
        format_transformation_rule(t["transformation_name"], t.get("location", "unspecified"), lang_key)
        for t in transformations
    ]
    transformation_block = "\n".join(transformation_lines)
    header_template = PROMPT_TEMPLATES[lang_key][template_index % len(PROMPT_TEMPLATES[lang_key])]
    prompt = header_template.format(
        transformation_block=transformation_block,
        code_str=code_str.strip()
    )
    return prompt


def read_json_file(file_path):
    """
    尝试以多种格式读取JSON文件
    支持: JSONL, 单个JSON对象, JSON数组
    """
    print(f"正在读取文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"文件为空: {file_path}")
    
    # 尝试作为完整JSON解析
    try:
        data = json.loads(content)
        print(f"✅ 成功解析为完整JSON，类型: {type(data).__name__}")
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]  # 单个对象包装成列表
        else:
            raise ValueError(f"不支持的JSON类型: {type(data)}")
            
    except json.JSONDecodeError:
        print("尝试作为JSONL格式解析...")
        
    # 尝试作为JSONL解析
    lines = content.split('\n')
    items = []
    valid_lines = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        try:
            item = json.loads(line)
            items.append(item)
            valid_lines += 1
        except json.JSONDecodeError as e:
            print(f"⚠️  行 {i+1} JSON解析错误: {e}")
            # 尝试修复末尾逗号问题
            if line.endswith(','):
                try:
                    item = json.loads(line[:-1])
                    items.append(item)
                    valid_lines += 1
                    print(f"✅ 修复了行 {i+1} 的末尾逗号")
                except:
                    print(f"❌ 无法修复行 {i+1}: {line[:50]}...")
            else:
                print(f"❌ 跳过行 {i+1}: {line[:50]}...")
    
    if valid_lines == 0:
        raise ValueError(f"没有找到有效的JSON数据: {file_path}")
    
    print(f"✅ 成功解析 {valid_lines} 行JSONL数据")
    return items


def generate_augmented_prompts_from_files(code_file, score_file, topk=3, language="java"):
    lang_key = normalize_language(language)

    try:
        # 读取代码文件
        print("=" * 50)
        code_items = read_json_file(code_file)
        code_map = {item["idx"]: item for item in code_items}
        print(f"加载了 {len(code_map)} 个代码样本")
        
        # 读取评分文件
        print("=" * 50)
        score_items = read_json_file(score_file)
        print(f"加载了 {len(score_items)} 个评分样本")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return []

    results = []
    processed = 0
    skipped = 0
    
    print("=" * 50)
    print("开始生成增强提示...")
    
    for entry in score_items:
        # 兼容不同的键名
        idx = entry.get("index") or entry.get("idx")
        if idx is None:
            print(f"⚠️  跳过条目，缺少index/idx字段: {entry}")
            skipped += 1
            continue
        
        patterns = entry.get("patterns", [])
        if not patterns:
            print(f"⚠️  跳过idx {idx}，没有patterns")
            skipped += 1
            continue
            
        sorted_patterns = sorted(patterns, key=lambda p: p.get("score", 0), reverse=True)
        top_patterns = sorted_patterns[:topk]

        code_entry = code_map.get(idx)
        if not code_entry:
            print(f"⚠️  跳过idx {idx}，找不到对应的代码")
            skipped += 1
            continue

        try:
            # 兼容不同的代码字段名
            code_str = code_entry.get("func") or code_entry.get("code") or code_entry.get("function")
            if not code_str:
                print(f"⚠️  跳过idx {idx}，找不到代码字段")
                skipped += 1
                continue
                
            prompt = generate_augmented_prompt(code_str, top_patterns, language=lang_key)
            results.append({
                "idx": idx,
                "prompt": prompt,
                "complexity_score": sum(p.get("score", 0) for p in top_patterns) / len(top_patterns) if top_patterns else 0,
                "num_transformations": len(top_patterns)
            })
            processed += 1
            
            if processed % 10 == 0:
                print(f"已处理 {processed} 个样本...")
                
        except Exception as e:
            print(f"❌ 处理idx {idx}时出错: {e}")
            skipped += 1

    print(f"处理完成: 成功 {processed} 个，跳过 {skipped} 个")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', type=str, required=True, help='代码文件路径(JSONL格式)')
    parser.add_argument('--score_file', type=str, required=True, help='评分文件路径(JSON/JSONL格式)')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    parser.add_argument('--topk', type=int, default=3, help='每个样本选择的top变换数量')
    parser.add_argument('--language', type=str, choices=["java", "cpp"], default="java", help='编程语言')
    args = parser.parse_args()

    print(f"参数配置:")
    print(f"  代码文件: {args.code_file}")
    print(f"  评分文件: {args.score_file}")
    print(f"  输出文件: {args.output_file}")
    print(f"  TopK: {args.topk}")
    print(f"  语言: {args.language}")
    print()

    try:
        prompts = generate_augmented_prompts_from_files(
            args.code_file, args.score_file, topk=args.topk, language=args.language
        )
        
        if prompts:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for item in prompts:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

            print("=" * 50)
            print(f"✅ 成功生成 {len(prompts)} 个提示，已保存到: {args.output_file}")
            
            avg_complexity = sum(p["complexity_score"] for p in prompts) / len(prompts)
            avg_transformations = sum(p["num_transformations"] for p in prompts) / len(prompts)
            print(f"📊 平均复杂度得分: {avg_complexity:.2f}")
            print(f"📊 平均变换数量: {avg_transformations:.2f}")
        else:
            print("❌ 没有生成任何提示")
            
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()