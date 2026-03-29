import json 
from collections import defaultdict
from typing import Dict, List, Set

def build_inverted_index(doc_tokens: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    index = defaultdict(set)
    for doc_id, tokens in doc_tokens.items():
        for term in set(tokens):
            index[term].add(doc_id)
    return dict(index)

def save_inverted_index(index: Dict[str, Set[str]], path: str) -> None:
    serializable = {term: sorted(list(doc_ids)) for term, doc_ids in index.items()}
    with open(path, "w", encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)
        
        
def load_inverted_index(path: str) -> Dict[str, Set[str]]:
    with open(path, "r", encoding='utf-8') as f:
        raw = json.load(f)
    return {term: set(doc_ids) for term, doc_ids in raw.items()}



