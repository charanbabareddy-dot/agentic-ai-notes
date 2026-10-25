"""
A safe calculator for agent tools.

Never use Python's eval() on text that comes from an LLM or a user: it can run any code.
This function parses the text and allows only numbers and + - * / ( ).
"""

import ast
import operator

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero is not allowed.")
        return _OPERATORS[type(node.op)](left, right)
    raise ValueError("Only numbers and + - * / ( ) are allowed.")


def calculate(expression: str) -> str:
    """Return the result as text, or a clear error message (never raises)."""
    if len(expression) > 200:
        return "Error: expression is too long."
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
        return str(round(result, 6))
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Error: {exc}"
