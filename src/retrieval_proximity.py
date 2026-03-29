from typing import Dict, List, Set


def parse_proximity_query(query: str):
    parts = query.strip().split()
    if len(parts) != 3 or not parts[2].startswith("/"):
        raise ValueError("Invalid proximity query format. Use: term1 term2 /k")
    t1, t2 = parts[0], parts[1]
    k = int(parts[2][1:])
    return t1, t2, k


def within_k(pos_list_a: List[int], pos_list_b: List[int], k: int) -> bool:
    """Check if any pair of positions has exactly k words between them (diff == k+1)."""
    target = k + 1
    i, j = 0, 0
    while i < len(pos_list_a) and j < len(pos_list_b):
        diff = abs(pos_list_a[i] - pos_list_b[j])
        if diff == target:
            return True
        if pos_list_a[i] < pos_list_b[j]:
            i += 1
        else:
            j += 1
    return False


def proximity_search(
    term1: str,
    term2: str,
    k: int,
    positional_index: Dict[str, Dict[str, List[int]]],
) -> Set[str]:
    result = set()

    postings1 = positional_index.get(term1, {})
    postings2 = positional_index.get(term2, {})

    common_docs = set(postings1.keys()) & set(postings2.keys())
    for doc_id in common_docs:
        if within_k(postings1[doc_id], postings2[doc_id], k):
            result.add(doc_id)

    return result