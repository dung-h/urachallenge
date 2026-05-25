from __future__ import annotations

import ast
import math
import re

from app.physics.formulas import K_COULOMB


_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def normalize_equation_expression(expression: str) -> str:
    text = expression.strip()
    text = text.replace("^", "**")
    text = text.replace("×", "*").replace("·", "*").replace("∙", "*")
    text = text.replace("π", "pi")
    text = text.replace("λ", "lam").replace("σ", "sigma").replace("ρ", "rho")
    text = re.sub(r"\b(?:ε₀|epsilon_0|epsilon0|eps0)\b", "eps0", text)
    text = text.replace("ε₀", "eps0")
    text = re.sub(
        r"([A-Za-z0-9_)\]])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)",
        lambda match: f"{match.group(1)}**{match.group(2).translate(_SUPERSCRIPT_TO_ASCII)}",
        text,
    )
    text = text.translate(_SUPERSCRIPT_TO_ASCII)
    text = re.sub(r"(?<=[0-9])(?=[A-Za-z_(])", "*", text)
    text = re.sub(r"(?<=[)\]])(?=[A-Za-z0-9_(])", "*", text)
    text = re.sub(r"(?<=pi)(?=eps0\b)", "*", text)
    text = re.sub(r"(?<=eps0)(?=[A-Za-z_(])", "*", text)
    text = re.sub(r"\*{2,}", "**", text)
    return text


def safe_eval_expression(expression: str, variables: dict[str, float]) -> float:
    normalized = normalize_equation_expression(expression)
    tree = ast.parse(normalized, mode="eval")
    allowed_names = {
        "k": K_COULOMB,
        "pi": math.pi,
        "sqrt": math.sqrt,
        "eps0": 8.854e-12,
        "epsilon_0": 8.854e-12,
        "epsilon0": 8.854e-12,
        **variables,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in allowed_names:
            value = allowed_names[node.id]
            if callable(value):
                raise ValueError("function name cannot be used as value")
            return float(value)
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
        if isinstance(node, ast.UnaryOp):
            value = eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("chained comparisons are not supported")
            left = eval_node(node.left)
            right = eval_node(node.comparators[0])
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                return 1.0 if left < right else 0.0
            if isinstance(op, ast.LtE):
                return 1.0 if left <= right else 0.0
            if isinstance(op, ast.Gt):
                return 1.0 if left > right else 0.0
            if isinstance(op, ast.GtE):
                return 1.0 if left >= right else 0.0
            if isinstance(op, ast.Eq):
                return 1.0 if left == right else 0.0
            if isinstance(op, ast.NotEq):
                return 1.0 if left != right else 0.0
            raise ValueError(f"unsupported comparison operator: {type(op).__name__}")
        if isinstance(node, ast.IfExp):
            test = eval_node(node.test)
            return eval_node(node.body) if test else eval_node(node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
            if len(node.args) != 1:
                raise ValueError("sqrt expects one argument")
            return math.sqrt(eval_node(node.args[0]))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    return eval_node(tree)
