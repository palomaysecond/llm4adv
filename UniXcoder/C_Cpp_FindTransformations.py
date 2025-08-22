"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: C_Cpp_FindTransformations.py.py
@time: 2025/6/25 21:49
"""
import numpy as np

def get_descendants(node):
    """递归返回所有后代节点"""
    descendants = []
    nodes_to_visit = [node]
    while nodes_to_visit:
        current = nodes_to_visit.pop()
        children = list(current.children)
        descendants.extend(children)
        nodes_to_visit.extend(children)
    return descendants


def AssessForToWhileConversion_new(node, source_code):
    """
    C/C++: 检查一个节点是否是 'for' 循环语句。
    这包括经典的 for(i=0; i<n; i++) 循环和C++11的 for-range 循环。
    """
    if node.type in ['for_statement', 'for_range_loop']:
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False

def AssessInlineLoopDeclaration_new(node, source_code):
    """
    C/C++: 检查一个节点是否是变量声明，并且紧跟着一个使用该变量的 for 循环。
    """
    if node.type != 'declaration':
        return False
    declared_vars = set()
    for declarator_node in node.children_by_field_name('declarator'):
        if declarator_node.type == 'init_declarator':
            value_node = declarator_node.child_by_field_name('value')
            if value_node is None:
                name_node = declarator_node.child_by_field_name('declarator')
                if name_node and name_node.type == 'identifier':
                    var_name = source_code[name_node.start_byte:name_node.end_byte]
                    declared_vars.add(var_name)
    if not declared_vars:
        return False
    for_node = node.next_named_sibling
    if for_node is None or for_node.type != 'for_statement':
        return False
    init_node = for_node.child_by_field_name('init')
    if init_node and init_node.type == 'expression_statement':
        if init_node.named_child_count > 0 and init_node.named_children[0].type == 'assignment_expression':
            assignment_expr = init_node.named_children[0]
            loop_var_node = assignment_expr.child_by_field_name('left')
            if loop_var_node and loop_var_node.type == 'identifier':
                loop_var_name = source_code[loop_var_node.start_byte:loop_var_node.end_byte]
                if loop_var_name in declared_vars:
                    body_node = for_node.child_by_field_name('body')
                    end_byte = body_node.start_byte if body_node else for_node.end_byte
                    end_point = body_node.start_point if body_node else for_node.end_point
                    return (True, node.start_byte, end_byte, node.start_point, end_point)
    return False

def AssessExtractLoopDeclaration_new(node, source_code):
    """
    C/C++: 检查一个节点是否是初始化部分为变量声明的 for 循环。
    """
    if node.type == 'for_statement':
        init_node = node.child_by_field_name('init')
        if init_node and init_node.type == 'declaration':
            body_node = node.child_by_field_name('body')
            if body_node:
                r_paren_node = node.child_by_field_name('condition').next_sibling
                while r_paren_node and r_paren_node.type != ')':
                    r_paren_node = r_paren_node.next_sibling
                if r_paren_node and r_paren_node.type == ')':
                    end_byte = r_paren_node.end_byte
                    return (True, node.start_byte, end_byte, node.start_point, (body_node.start_point[0], body_node.start_point[1]))
    return False

def AssessIfElseBranchSwap_new(node, source_code):
    """
    C/C++: 检查一个节点是否是可以交换分支的if-else语句。
    """
    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        alternative_node = node.child_by_field_name('alternative')
        if not (consequence_node and alternative_node):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.type == 'parenthesized_expression':
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['==', '!=']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessElseIfToNestedIf_new(node, source_code):
    """
    C/C++: 检查一个节点是否是带代码块的 'else if' 语句。
    """
    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'compound_statement'):
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

def AssessNestedIfToElseIf_new(node, source_code):
    """
    C/C++: 检查 'else { if (...) {} }' 结构。
    """
    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'compound_statement'):
        return False
    parent_block = node.parent
    if not parent_block or parent_block.type != 'compound_statement' or parent_block.named_child_count != 1:
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

def AssessUnwrapRedundantBlock_new(node, source_code):
    """
    C/C++: 检查带花括号的if语句。
    """
    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type == 'compound_statement':
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessWrapStatementInBlock_new(node, source_code):
    """
    C/C++: 检查不带花括号的if语句。
    """
    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type != 'compound_statement':
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessSplitCompoundCondition_new(node, source_code):
    """
    C/C++: 检查条件为逻辑与/或的if语句。
    """
    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        if not (consequence_node and consequence_node.type == 'compound_statement'):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.type == 'parenthesized_expression':
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['&&', '||']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessAddRedundantStatement_new(node, source_code):
    """
    C/C++: 检查局部变量声明语句。
    """
    if node.type == 'declaration':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessWrapWithConstantCondition_new(node, source_code):
    """
    C/C++: 检查独立的赋值语句。
    """
    if node.type != 'expression_statement':
        return False
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    if node.parent and node.parent.type == 'compound_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessExtractSubexpression_new(node, source_code):
    """
    C/C++: 检查赋值语句右侧是否包含复杂部分。
    """
    if not (node.type == 'expression_statement' and node.parent and node.parent.type == 'compound_statement'):
        return False
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    assignment_node = node.named_children[0]
    rhs_node = assignment_node.child_by_field_name('right')
    if not (rhs_node and rhs_node.type == 'binary_expression'):
        return False
    left_operand = rhs_node.child_by_field_name('left')
    right_operand = rhs_node.child_by_field_name('right')
    if (left_operand and left_operand.type in ['binary_expression', 'call_expression']) or \
       (right_operand and right_operand.type in ['binary_expression', 'call_expression']):
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessSwitchToIfElse_new(node, source_code):
    """
    C/C++: 检查switch结构。
    """
    if node.type == 'switch_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessReturnViaTempVariable_new(node, source_code):
    """
    C/C++: 检查返回字面量的return语句。
    """
    if node.type == 'return_statement':
        if node.named_child_count > 0:
            return_value_node = node.named_children[0]
            literal_types = {'number_literal', 'string_literal', 'char_literal', 'true', 'false', 'null'}
            if return_value_node.type in literal_types:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessPromoteIntToLong_new(node, source_code):
    """
    C/C++: 识别可以从 int 提升到 long 的整数字面量声明。
    """
    if node.type == 'declaration':
        type_node = node.child_by_field_name('type')
        if type_node and type_node.type == 'primitive_type':
            type_text = source_code[type_node.start_byte:type_node.end_byte]
            if type_text == 'int':
                for declarator in node.children_by_field_name('declarator'):
                    if declarator.type == 'init_declarator':
                        value_node = declarator.child_by_field_name('value')
                        if value_node and value_node.type == 'number_literal':
                            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessPromoteFloatToDouble_new(node, source_code):
    """
    C/C++: 识别可以从 float 提升到 double 的浮点字面量声明。
    """
    if node.type == 'declaration':
        type_node = node.child_by_field_name('type')
        if type_node and type_node.type == 'primitive_type':
            type_text = source_code[type_node.start_byte:type_node.end_byte]
            if type_text == 'float':
                for declarator in node.children_by_field_name('declarator'):
                     if declarator.type == 'init_declarator':
                        value_node = declarator.child_by_field_name('value')
                        if value_node and value_node.type == 'number_literal':
                            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessAddUnusedParameter_new(node, source_code):
    """
        识别一个函数或方法声明/定义，用于后续添加未使用的参数。
        对于 C/C++，这主要针对函数定义和函数声明（原型）。
        例如: void myFunc(int a) { ... } 或者 void myFunc(int a);
        """
    # 情况1: 函数定义 (例如, void myFunc(...) { ... })
    if node.type == 'function_definition':
        # 函数体是一个 'compound_statement' 节点。
        # 在 C++ 的 tree-sitter 语法中，它是一个命名字段 'body'；而在C中，它通常是最后一个子节点。
        body_node = node.child_by_field_name('body')
        if not body_node:
            # 兼容C的语法：查找 'compound_statement' 类型的子节点
            for child in reversed(node.children):
                if child.type == 'compound_statement':
                    body_node = child
                    break

        if body_node:
            # 变换范围是从函数开始到函数体 '{' 开始之前
            # 这可以完整地包括函数签名、C++的 noexcept/const 等修饰符
            return (True, node.start_byte, body_node.start_byte, node.start_point, body_node.start_point)
        else:
            # 一个函数定义应该有函数体。如果没有，可能是C++的 '= delete' 或 '= default'。
            # 在这种情况下，我们将整个节点作为目标范围。
            # 例如: MyClass() = default;
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    # 情况2: 函数声明/原型 (例如, void myFunc(...);)
    # 它们通常是 'declaration' 或 'field_declaration' (用于 C++ 类成员) 节点。
    if node.type in ['declaration', 'field_declaration']:
        # 为了确认这是一个函数声明，而不是变量声明，我们检查它是否包含一个 'function_declarator'。
        # 同时，我们需要排除函数指针变量的声明，例如 'int (*p)(void);'。
        # 函数声明的 'function_declarator' 不会包含 'parenthesized_declarator'。
        q = list(node.children)
        while q:
            child = q.pop(0)
            if child.type == 'function_declarator':
                # 检查是否存在 parenthesized_declarator，以排除函数指针
                is_function_pointer = any(
                    # d.type == 'parenthesized_declarator' for d in child.descendants
                    d.type == 'parenthesized_declarator' for d in get_descendants(child)
                )
                if not is_function_pointer:
                    # 这是一个函数原型，整个节点都是目标
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

            # 继续在子节点中查找
            q.extend(child.children)

    return False


def AssessRefactorOutputAPI_new(node, source_code):
    """
    C/C++: 检查是否是 printf 或 std::cout 语句。
    """
    if node.type == 'call_expression':
        func_node = node.child_by_field_name('function')
        if func_node:
            # C: printf
            if func_node.type == 'identifier':
                func_name = source_code[func_node.start_byte:func_node.end_byte]
                if func_name == 'printf':
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            # C++: std::cout << (represented as a call_expression with `<<` operator)
            elif func_node.type == 'field_expression':
                 if 'cout' in source_code[func_node.start_byte:func_node.end_byte]:
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessRenameVariable_new(node, source_code):
    """
    C/C++: 检查变量或参数声明中的 'identifier'。
    """
    if node.type != 'identifier':
        return False
    parent = node.parent
    if not parent:
        return False
    # 上下文 A: 局部变量或字段声明
    # declaration -> init_declarator -> identifier
    if parent.type == 'init_declarator' and parent.child_by_field_name('declarator') == node:
        grandparent = parent.parent
        if grandparent and grandparent.type == 'declaration':
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    # 上下文 B: 函数参数声明
    # parameter_declaration -> identifier
    elif parent.type == 'parameter_declaration' and parent.child_by_field_name('declarator') == node:
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessRenameClassAndMethod_new(node, source_code):
    """
    C/C++: 检查类/结构体声明或函数定义。
    """
    # C++ 类/结构体声明
    if node.type in ['class_specifier', 'struct_specifier']:
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    # C/C++ 函数定义
    elif node.type == 'function_definition':
        declarator_node = node.child_by_field_name('declarator')
        if declarator_node:
            # 循环找到最内层的 declarator 以获取名字
            while declarator_node.child_by_field_name('declarator'):
                declarator_node = declarator_node.child_by_field_name('declarator')
            name_node = declarator_node.child_by_field_name('declarator')
            if name_node and name_node.type == 'identifier':
                 return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    return False
def AssessWhileToForRefactoring_new(node, source_code):
    """
    C/C++: 检查一个节点是否是 'while' 循环语句。
    """
    if node.type == 'while_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessDoWhileToWhileConversion_new(node, source_code):
    """
    C/C++: 检查一个节点是否是 'do-while' 循环语句。
    例如: do { ... } while(condition);
    """
    if node.type == 'do_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessNegateWithReversedOperator_new(node, source_code):
    """
    C/C++: 检查一个节点是否是一个二元比较表达式。
    例如: x < y, a >= b, c == d 等。
    """
    if node.type == 'binary_expression':
        comparison_operators = {'<', '>', '<=', '>=', '==', '!='}
        for child in node.children:
            if child.type in comparison_operators:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessExpandCompoundAssign_new(node, source_code):
    """
    C/C++: 检查一个节点是否是复合赋值表达式，且该表达式是独立语句或for循环的更新部分。
    例如: x += 1, y -= 2, z *= 3; 或 for(...; ...; i += 2)
    """
    if node.type == 'assignment_expression':
        compound_operators = {
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='
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


def AssessExpandUnaryOP_new(node, source_code):
    """
    C/C++: 检查一个节点是否是一元更新表达式，且该表达式是独立语句或for循环的更新部分。
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
