import ast
import numpy as np


def analyze_var_names(code):
    # Анализ имен переменных
    tree = ast.parse(code)
    variables = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    name_len = [len(name) for name in variables]
    avg_name_len = np.mean(name_len) if name_len else 0

    return avg_name_len > 5



