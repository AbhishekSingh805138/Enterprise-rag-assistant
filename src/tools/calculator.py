"""Safe calculator tool for arithmetic expressions.

Uses Python's ast module to safely evaluate mathematical expressions
without executing arbitrary code. Supports basic arithmetic, powers,
and common math operations.
"""
from __future__ import annotations

import ast
import logging
import math
import operator

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Allowed binary operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Safe math constants and functions
_SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node with only allowed operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and right > 100:
            raise ValueError("Exponent too large (max 100)")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("Division by zero")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            val = _SAFE_NAMES[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' must be called as a function, e.g. {node.id}(...)")
            return float(val)
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            func = _SAFE_NAMES[node.func.id]
            if not callable(func):
                raise ValueError(f"'{node.func.id}' is not callable")
            args = [_safe_eval(arg) for arg in node.args]
            return float(func(*args))
        raise ValueError("Unsupported function call")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, //, %, ** (power), parentheses,
    and functions: abs, round, min, max, sqrt, log, log10, ceil, floor.
    Constants: pi, e.

    Examples: "47.3 * 1.12", "sqrt(144)", "round(3.14159, 2)"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Format nicely: avoid trailing .0 for integers
        if result == int(result):
            formatted = str(int(result))
        else:
            formatted = f"{result:.6g}"
        logger.info("Calculator: %s = %s", expression.strip(), formatted)
        return formatted
    except (SyntaxError, ValueError, TypeError, OverflowError) as e:
        logger.warning("Calculator error: %s — %s", expression.strip(), e)
        return f"Error: {e}"
