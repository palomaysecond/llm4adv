"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: Java_FindTransformations.py
@time: 2025/6/14 14:08
"""
import numpy as np
import json
import argparse
from GetAST import generateASt
from ImportanceAnalyze_unix import load_preprocessed_data


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


def walk_tree(node):
    yield node
    for child in node.children:
        yield from walk_tree(child)


def AssessForToWhileConversion(node, source_code):
    """
    检查一个节点是否是 'for' 循环语句。
    这包括经典的 for(i=0; i<n; i++) 循环和增强型for循环。
    """
    if node.type in ['for_statement', 'enhanced_for_statement']:
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessWhileToForRefactoring(node, source_code):
    """
    检查一个节点是否是 'while' 循环语句。
    """
    if node.type == 'while_statement':
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessDoWhileToWhileConversion(node, source_code):
    """
    检查一个节点是否是 'do-while' 循环语句。
    例如: do { ... } while(condition);
    """
    if node.type == 'do_statement':
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessInlineLoopDeclaration(node, source_code):
    """
    检查一个节点是否是变量声明，并且紧跟着一个使用该变量的 for 循环。
    例如: int i; for (i = 0; ...)
    如果匹配，返回的范围将包含变量声明和整个 for 循环。
    """
    # 1. 确保当前节点是Java的局部变量声明
    if node.type != 'local_variable_declaration':
        return False
    # 2. 从声明中找出所有未初始化的变量名
    declared_vars = set()
    # 遍历声明中的所有 declarator (例如, 在 int i, j; 中有两个)
    for declarator_node in node.children:
        if declarator_node.type == 'variable_declarator':
            # 检查变量是否被初始化。如果 'value' 字段不存在，则未初始化。
            value_node = declarator_node.child_by_field_name('value')
            if value_node is None:
                # 获取变量名
                name_node = declarator_node.child_by_field_name('name')
                if name_node and name_node.type == 'identifier':
                    var_name = source_code[name_node.start_byte:name_node.end_byte]
                    declared_vars.add(var_name)
    
    if not declared_vars:
        return False
    
    # 3. 检查紧随其后的兄弟节点是否为 for 循环
    for_node = node.next_named_sibling
    if for_node is None or for_node.type != 'for_statement':
        return False

    # 4. 检查 for 循环是否在初始化部分使用了我们找到的变量
    init_node = for_node.child_by_field_name('init')
    # 初始化部分通常是包含赋值的表达式语句
    if init_node and init_node.type == 'expression_statement':
        if init_node.named_child_count > 0 and init_node.named_children[0].type == 'assignment_expression':
            assignment_expr = init_node.named_children[0]
            # 赋值表达式的左侧应该是我们之前声明的变量
            loop_var_node = assignment_expr.child_by_field_name('left')
            if loop_var_node and loop_var_node.type == 'identifier':
                loop_var_name = source_code[loop_var_node.start_byte:loop_var_node.end_byte]
                
                # 5. 如果变量名匹配，则返回覆盖两个语句的范围
                if loop_var_name in declared_vars:
                    start_byte = node.start_byte
                    start_point = node.start_point
                    # 将范围的结束位置设置为循环体开始之前
                    body_node = for_node.child_by_field_name('body')
                    if body_node:
                        end_byte = body_node.start_byte
                        end_point = body_node.start_point
                    else:
                        # Fallback for unusual cases like `for(;;)` without a final semicolon
                        end_byte = for_node.end_byte
                        end_point = for_node.end_point
                    return (True, start_byte, end_byte, start_point, end_point)

    return False

def AssessExtractLoopDeclaration(node, source_code):
    """
    检查一个节点是否是初始化部分为变量声明的 for 循环。
    例如: for (int i = 0; ...)
    """
    if node.type == 'for_statement':
        init_node = node.child_by_field_name('init')
        # 检查初始化子节点的类型是否为 'local_variable_declaration'
        if init_node and init_node.type == 'local_variable_declaration':
            # 精确范围：从 for 关键字开始，到循环体 { 之前结束
            body_node = node.child_by_field_name('body')
            if body_node:
                end_byte = body_node.start_byte
                r_paren_node = node.child(node.child_count - 2)
                if r_paren_node and r_paren_node.type == ')':
                    end_byte = r_paren_node.end_byte
            return (True, node.start_byte, end_byte, node.start_point, (body_node.start_point[0], body_node.start_point[1]))
    return False


def AssessIfElseBranchSwap(node, source_code):
    """
    检查一个节点是否是条件为 '==' 或 '!=' 的二元表达式的if语句。
    例如: if (x == 10) 或 if (y != null)
    """
    # if node.type == 'if_statement':
    #     condition_node = node.child_by_field_name('condition')
    #     if condition_node and condition_node.named_child_count > 0:
    #         actual_condition = condition_node.named_children[0]
    #         if actual_condition.type == 'binary_expression':
    #             for child in actual_condition.children:
    #                 if child.type in ['==', '!=']:
    #                     return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    # return False
    """
    检查一个节点是否是可以交换分支的if-else语句。
    范围只包含if语句的条件 '(...)' 部分。
    """
    if node.type == 'if_statement':
        # 必须同时有 then (consequence) 和 else (alternative) 分支
        consequence_node = node.child_by_field_name('consequence')
        alternative_node = node.child_by_field_name('alternative')
        if not (consequence_node and alternative_node):
            return False

        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.named_child_count > 0:
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['==', '!=']:
                        # 范围就是 '(...)' 括号表达式
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False


def AssessElseIfToNestedIf(node, source_code):
    """
    检查一个节点是否是带代码块的 'else if' 语句。
    例如: else if (condition) { ... }
    如果匹配，它不只返回 True，而是返回一个元组: (True, start_byte, end_byte)
    其中字节范围包含了 'else' 关键字，以提取 'else if (...) { ... }' 的完整文本。
    """
    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'block'):
        return False
    parent = node.parent
    if not parent or parent.type != 'if_statement':
        return False
    alternative_node = parent.child_by_field_name('alternative')
    if not (alternative_node and alternative_node.id == node.id):
        return False
    node_index = -1
    for i, child in enumerate(parent.children):
        if child.id == node.id:
            node_index = i
            break
    if node_index > 0:
        prev_sibling = parent.children[node_index - 1]
        if prev_sibling.type == 'else':
            return (True, prev_sibling.start_byte, node.end_byte, prev_sibling.start_point, node.end_point)
    return False


def AssessNestedIfToElseIf(node, source_code):
    """
    检查一个节点是否是 'else { if (...) {} }' 结构中的那个内层if语句。
    例如: else { if (condition) { ... } }
    如果匹配，它将返回一个元组 (True, start_byte, end_byte)
    其中字节范围覆盖了从 'else' 关键字到整个内部 if 语句的结束。
    """
    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'block'):
        return False
    parent_block = node.parent
    if not parent_block or parent_block.type != 'block' or parent_block.named_child_count != 1:
        return False
    grandparent_if = parent_block.parent
    if not grandparent_if or grandparent_if.type != 'if_statement':
        return False
    alternative_node = grandparent_if.child_by_field_name('alternative')
    if not (alternative_node and alternative_node.id == parent_block.id):
        return False
    node_index = -1
    for i, child in enumerate(grandparent_if.children):
        if child.id == parent_block.id:
            node_index = i
            break
    if node_index > 0:
        prev_sibling = grandparent_if.children[node_index - 1]
        if prev_sibling.type == 'else':
            return (True, prev_sibling.start_byte, parent_block.end_byte, prev_sibling.start_point, parent_block.end_point)
    return False

def AssessUnwrapRedundantBlock(node, source_code):
    """
    检查一个节点是否是带代码块 (braces) 的 if 语句。
    例如: if (x) { a(); }
    """
    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type == 'block':
            # 范围就是这个 block
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessWrapStatementInBlock(node, source_code):
    """
    检查一个节点是否是不带代码块 (braces) 的 if 语句。
    例如: if (x) a();
    """
    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type != 'block':
            # 范围就是这个无{}块状的执行体
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False


def AssessSplitCompoundCondition(node, source_code):
    """
    检查一个节点是否是 if 语句，其条件是逻辑与(&&)、逻辑或(||)表达式，并且带有一个代码块。
    例如: if (a && b) { ... }
    """
    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        if not (consequence_node and consequence_node.type == 'block'):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.named_child_count > 0:
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['&&', '||']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessAddRedundantStatement(node, source_code):
    """
    检查一个节点是否是局部变量声明语句。
    例如: float x = 1; 或 int a, b;
    """
    if node.type == 'local_variable_declaration':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessWrapWithConstantCondition(node, source_code):
    """
    检查一个节点是否是独立的赋值语句。
    例如: x = y + 1;
    这会排除 for 循环头等非独立语句中的赋值。
    """
    # 1. 节点本身需要是一个表达式语句
    if node.type != 'expression_statement':
        return False

    # 2. 该表达式语句的直接子节点是一个赋值表达式
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False

    # 3. 父节点必须是一个 block，以确保它是独立的语句
    # 这可以排除 for(i=0;...) 中的赋值表达式
    if node.parent and node.parent.type == 'block':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False
    

def AssessExtractSubexpression(node, source_code):
    """
    检查一个赋值语句的右侧是否包含可以被提取的复杂部分。
    这通常意味着右侧的表达式中包含了嵌套的二元表达式或方法调用。
    例如: z = a * b + c; (可提取 a * b)
         x = y + someFunc(); (可提取 someFunc())
    不匹配: x = y / 2; (没有可提取的部分)
    """
    # 1. 必须是独立的表达式语句
    if not (node.type == 'expression_statement' and node.parent and node.parent.type == 'block'):
        return False

    # 2. 表达式的主体必须是赋值表达式
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    
    assignment_node = node.named_children[0]

    # 3. 赋值表达式的右侧必须是二元表达式
    rhs_node = assignment_node.child_by_field_name('right')
    if not (rhs_node and rhs_node.type == 'binary_expression'):
        return False

    # 4. 检查二元表达式的左右操作数，看它们是否是可提取的复杂表达式
    left_operand = rhs_node.child_by_field_name('left')
    right_operand = rhs_node.child_by_field_name('right')

    # 如果左操作数或右操作数本身也是一个二元表达式或方法调用，则认为它可以被提取
    if (left_operand and left_operand.type in ['binary_expression', 'method_invocation']) or \
       (right_operand and right_operand.type in ['binary_expression', 'method_invocation']):
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False


def AssessSwitchToIfElse(node, source_code):
    """
    检查一个节点是否是switch结构
    switch_expression 是自 Java 14 引入的现代化 switch，它可以返回一个值，语法更简洁
    Java中更传统、更常见的 switch 是 switch_statement，它是一个语句，不返回值
    """
    if node.type in ['switch_expression', 'switch_statement']:
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessReturnViaTempVariable(node, source_code):
    """
    检查一个节点是否是返回一个字面量/常量的 return 语句。
    例如: return 1 return "error";, return true;, return null;
    """
    if node.type == 'return_statement':
        if node.named_child_count > 0:
            return_value_node = node.named_children[0]
            literal_types = {
                'decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal',
                'binary_integer_literal', 'decimal_floating_point_literal', 'hex_floating_point_literal',
                'string_literal', 'character_literal', 'boolean_literal', 'null_literal',
            }
            if return_value_node.type in literal_types:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessNegateWithReversedOperator(node, source_code):
    """
    检查一个节点是否是一个二元比较表达式。
    例如: x < y, a >= b, c == d 等。
    """
    if node.type == 'binary_expression':
        comparison_operators = {'<', '>', '<=', '>=', '==', '!='}
        for child in node.children:
            if child.type in comparison_operators:
                return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessExpandCompoundAssign(node, source_code):
    """
    检查一个节点是否是复合赋值表达式，且该表达式是独立语句或for循环的更新部分。
    例如: x += 1, y -= 2, z *= 3 或 for(...; ...; i += 2)
    """
    if node.type == 'assignment_expression':
        compound_operators = {
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='
        }
        has_compound_op = any(child.type in compound_operators for child in node.children)
        
        if has_compound_op:
            parent = node.parent
            if not parent:
                return False

            # 情况1: 作为一个独立的语句存在
            if parent.type == 'expression_statement':
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

            # 情况2: 作为 for 循环的 update 部分存在
            if parent.type == 'for_statement' and parent.child_by_field_name('update') == node:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            
    return False


def AssessExpandUnaryOP(node, source_code):
    """
    检查一个节点是否是一元更新表达式，且该表达式是独立语句或for循环的更新部分。
    例如: i++;, --j; 或者 for(...; ...; i++)
    """
    if node.type == 'update_expression':
        parent = node.parent
        if not parent:
            return False

        # 情况1: 作为一个独立的语句存在
        if parent.type == 'expression_statement':
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

        # 情况2: 作为 for 循环的 update 部分存在
        if parent.type == 'for_statement' and parent.child_by_field_name('update') == node:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            
    return False


def AssessPromoteIntToLong(node, source_code):
    """
    识别可以从 int 提升到 long 的整数字面量声明。
    例如: int x = 10;
    """
    if node.type == 'local_variable_declaration':
        type_node = node.child_by_field_name('type')
        if type_node:
            type_text = source_code.encode('utf-8')[type_node.start_byte:type_node.end_byte]
            if type_text == b'int':
                for declarator in node.children_by_field_name('declarator'):
                    value_node = declarator.child_by_field_name('value')
                    if value_node and 'literal' in value_node.type:
                        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessPromoteFloatToDouble(node, source_code):
    """
    识别可以从 float 提升到 double 的浮点字面量声明。
    例如: float f = 0.0f;
    """
    if node.type == 'local_variable_declaration':
        type_node = node.child_by_field_name('type')
        if type_node:
            type_text = source_code.encode('utf-8')[type_node.start_byte:type_node.end_byte]
            if type_text == b'float':
                for declarator in node.children_by_field_name('declarator'):
                    value_node = declarator.child_by_field_name('value')
                    if value_node and 'literal' in value_node.type:
                        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessAddUnusedParameter(node, source_code):
    """
    识别一个方法或构造函数声明，用于后续添加未使用的参数。
    此变换的目标是整个方法头（从修饰符到方法体'{'之前），包括参数列表和throws子句。
    例如: public void myMethod(int a) throws Exception 或 public MyClass(int a)
    """
    # 1. 检查节点是否为方法或构造函数声明
    if node.type in ['method_declaration', 'constructor_declaration']:

        # 2. 找到方法体节点 ('block' node or a ';')
        body_node = node.child_by_field_name('body')

        # 3. 确定范围
        start_byte = node.start_byte
        start_point = node.start_point

        if body_node:
            # 如果有方法体，范围的结束点就在方法体开始之前
            end_byte = body_node.start_byte
            end_point = body_node.start_point
        else:
            # 如果没有方法体 (例如 abstract 方法或 interface 方法)
            # 范围就是整个节点
            end_byte = node.end_byte
            end_point = node.end_point

        # 返回成功和精确的范围
        return (True, start_byte, end_byte, start_point, end_point)

    # 4. 如果不是方法或构造函数声明，返回False
    return False


def AssessRefactorOutputAPI(node, source_code):
    """
    检查一个节点是否是 'System.out.println(...)' 语句。
    """
    if node.type != 'method_invocation':
        return False
    name_node = node.child_by_field_name('name')
    if not name_node:
        return False
    method_name = source_code[name_node.start_byte:name_node.end_byte]
    if method_name != 'println':
        return False
    object_node = node.child_by_field_name('object')
    if not object_node:
        return False
    object_name = source_code[object_node.start_byte:object_node.end_byte]
    if object_name == 'System.out':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessRenameVariable(node, source_code):
    """
        函数在最细粒度的 'identifier' 节点上触发，然后检查其上下文
        来判断这是否一个可重命名的声明（字段声明(private int x, y)、方法参数声明()、局部变量声明(int a, b = 1)）。
        可以处理 'int a, b;' 的情况，因为它会为 'a' 和 'b' 单独触发。
        返回 (True, start_byte, end_byte, start_point, end_point) 或 False。
    """
    # 1. 函数只在 'identifier' 节点上工作
    if node.type != 'identifier':
        return False

    parent = node.parent
    if not parent:
        return False

    # 2. 检查上下文：这个标识符是否是一个声明的名字？

    # 上下文 A: 局部变量或字段声明
    # AST 结构: ..._declaration -> variable_declarator -> identifier
    if parent.type == 'variable_declarator':
        grandparent = parent.parent
        # 确保它是一个局部变量或字段，并且我们捕获的是名字节点
        if grandparent and grandparent.type in ['local_variable_declaration', 'field_declaration']:
            if parent.child_by_field_name('name') == node:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    # 上下文 B: 方法参数声明
    # AST 结构: formal_parameter -> identifier
    elif parent.type == 'formal_parameter':
        if parent.child_by_field_name('name') == node:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    # 3. 如果以上上下文都不匹配，则这不是一个可重命名的声明
    return False

def AssessRenameClassAndMethod(node, source_code):
    """
        检查一个节点是否是类声明。
        例如: class MyClass { ... }

        检查一个节点是否是方法声明。
        例如: public void myMethod() { ... }
        """
    if node.type == 'class_declaration':
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    elif node.type == 'method_declaration':
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    return False