import re
from typing import List

OPERATORS = {"AND", "OR", "NOT"}
PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}


def tokenize_boolean_query(query: str) -> List[str]:
    # Captures parentheses and words
    raw = re.findall(r"\(|\)|[A-Za-z0-9_]+", query)
    tokens = []
    for t in raw:
        up = t.upper()
        tokens.append(up if up in OPERATORS else t)

    # Insert implicit AND between adjacent operands (e.g. "Hillary Clinton" -> "Hillary AND Clinton")
    result = []
    for i, tok in enumerate(tokens):
        result.append(tok)
        if i < len(tokens) - 1:
            curr_is_operand = tok not in OPERATORS and tok not in ("(", ")")
            next_is_operand = tokens[i + 1] not in OPERATORS and tokens[i + 1] not in ("(", ")")
            curr_is_close = tok == ")"
            next_is_open = tokens[i + 1] == "("
            if (curr_is_operand or curr_is_close) and (next_is_operand or next_is_open):
                result.append("AND")
    return result


def infix_to_postfix(tokens: List[str]) -> List[str]:
    output = []
    stack = []

    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()  # pop '('
        elif tok in OPERATORS:
            while stack and stack[-1] in OPERATORS and PRECEDENCE[stack[-1]] >= PRECEDENCE[tok]:
                output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(tok)

    while stack:
        output.append(stack.pop())

    return output