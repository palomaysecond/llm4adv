"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: GetAST.py
@time: 2025/5/26 11:00
"""
from tree_sitter import Parser
from tree_sitter import Tree
from tree_sitter import Language

Language.build_library(
    'build/my-languages.so',
    [
        'vendor/tree-sitter-python',
        'vendor/tree-sitter-java',
        'vendor/tree-sitter-c',
        'vendor/tree-sitter-cpp',
    ]
)

# .so 文件是 Linux 系统下
# JAVA_LANGUAGE=Language('build/my-languages.so','java')
# PYTHON_LANGUAGE=Language('build/my-languages.so','python')
# C_LANGUAGE = Language('build/my-languages.so', 'c')
# CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')

JAVA_LANGUAGE=Language('build/my-languages.so','java')
PYTHON_LANGUAGE=Language('build/my-languages.so','python')
C_LANGUAGE = Language('build/my-languages.so', 'c')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')


parser=Parser()


def generateASt(code,language):  # code是要解析的源代码，language是编程语言类型
    if language=='java':
        parser.set_language(JAVA_LANGUAGE)  # 根据指定的编程语言设置相应的解析器
    elif language=='python':
        parser.set_language(PYTHON_LANGUAGE)
    elif language=='c':
        parser.set_language(C_LANGUAGE)
    elif language=='cpp':
        parser.set_language(CPP_LANGUAGE)
    else:
        print('--wrong langauge--')
        return 0
    tree=parser.parse(bytes(code,encoding='utf-8'))  # 将源代码转换为UTF-8编码的字节格式，然后使用解析器解析代码生成语法树
    root_node=tree.root_node  # 返回语法树的根节点
    return root_node
