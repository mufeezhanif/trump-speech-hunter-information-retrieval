import json 
from collections import defaultdict
from typing import Dict, List, Set, Callable

from src.preprocessing import tokenize


def build_positional_index(
    docs: Dict[str, str],
    stopwords: Set[str],
    stem_fn: Callable[[str], str],
) -> Dict[str, Dict[str, List[int]]]:
    """Build positional index using original token positions."""
    index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for doc_id, text in docs.items():
        raw_tokens = tokenize(text)
        for pos, tok in enumerate(raw_tokens):
            if tok in stopwords:
                continue
            stemmed = stem_fn(tok)
            index[stemmed][doc_id].append(pos)

    return {term: dict(postings) for term, postings in index.items()}

def save_positional_index(index: Dict[str, Dict[str, List[int]]], path: str)-> None:
    with open(path, "w", encoding='utf-8') as f:
        json.dump(index, f, indent=2)
        

def load_positional_index(path: str) -> Dict[str, Dict[str, List[int]]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
