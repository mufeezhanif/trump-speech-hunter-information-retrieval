from typing import Dict, Set, List

def get_postings(term: str, inverted_index: Dict[str, Set[str]])-> Set[str]:
    return inverted_index.get(term, set())

def eval_simple_boolean(query_tokens: List[str], inverted_index: Dict[str, Set[str]], all_docs: Set[str])-> Set[str]:
    if len(query_tokens) == 2 and query_tokens[0]=='NOT':
        term_docs = get_postings(query_tokens[1], inverted_index)
        return all_docs - term_docs
    
    if len(query_tokens) == 3:
        left, op, right = query_tokens
        a = get_postings(left, inverted_index)
        b = get_postings(right, inverted_index)
        
        if op == "AND":
            return a & b
        if op == "OR":
            return a | b
        
    if len(query_tokens) == 5:
        first = eval_simple_boolean(query_tokens[:3], inverted_index, all_docs)
        second_term = query_tokens[4]
        op = query_tokens[3]
        second = get_postings(second_term, inverted_index)
        return first & second if op == "AND" else first | second
    
    raise ValueError("Unsupported simple query format")


def eval_postfix(postfix: List[str], inverted_index: Dict[str, Set[str]], all_docs: Set[str], normalize_term_fn) -> Set[str]:
    stack = []

    for tok in postfix:
        if tok == "NOT":
            a = stack.pop()
            stack.append(all_docs - a)
        elif tok in {"AND", "OR"}:
            right = stack.pop()
            left = stack.pop()
            stack.append(left & right if tok == "AND" else left | right)
        else:
            term = normalize_term_fn(tok)
            stack.append(inverted_index.get(term, set()))

    if len(stack) != 1:
        raise ValueError("Invalid query")
    return stack[0]