import os
import time

from src.io_utils import load_documents
from src.preprocessing import load_stopwords, preprocess_text, porter_stem_word, basic_stem_word
from src.index_inverted import build_inverted_index, save_inverted_index, load_inverted_index
from src.index_positional import build_positional_index, save_positional_index, load_positional_index
from src.query_parser import tokenize_boolean_query, infix_to_postfix
from src.retrieval_boolean import eval_postfix
from src.retrieval_proximity import parse_proximity_query, proximity_search

DATA_DIR = "data/Trump Speechs"
STOPWORD_PATH = "data/Stopword-List.txt"
INVERTED_PATH = "indexes/inverted_index.json"
POSITIONAL_PATH = "indexes/positional_index.json"
STEM_MODE = "porter"


def normalize_query_term(term, stopwords):
    toks = preprocess_text(term, stopwords, stem_mode=STEM_MODE)
    return toks[0] if toks else ""


def print_result(name, result):
    print("=" * 60)
    print(f"Query: {name}")
    print(f"Count: {len(result)}")
    print(f"Docs:  {sorted(result, key=lambda x: int(x))}")
    print()


def run_all_tests():
    docs = load_documents(DATA_DIR)
    stopwords = load_stopwords(STOPWORD_PATH)

    # Build or load indexes
    if not os.path.exists(INVERTED_PATH) or not os.path.exists(POSITIONAL_PATH):
        print("Building indexes...")
        t0 = time.perf_counter()
        doc_tokens = {doc_id: preprocess_text(text, stopwords, stem_mode=STEM_MODE) for doc_id, text in docs.items()}
        inverted = build_inverted_index(doc_tokens)
        stem_fn = porter_stem_word if STEM_MODE == "porter" else basic_stem_word
        positional = build_positional_index(docs, stopwords, stem_fn)
        os.makedirs("indexes", exist_ok=True)
        save_inverted_index(inverted, INVERTED_PATH)
        save_positional_index(positional, POSITIONAL_PATH)
        print(f"Index build time: {(time.perf_counter() - t0) * 1000:.2f} ms")
    else:
        print("Loading indexes from disk...")
        inverted = load_inverted_index(INVERTED_PATH)
        positional = load_positional_index(POSITIONAL_PATH)

    all_docs = set(docs.keys())
    norm_fn = lambda t: normalize_query_term(t, stopwords)

    # Simple Boolean
    boolean_queries = [
        "actions AND wanted",
        "united OR plane",
        "NOT hammer",
        "pakistan OR afganistan OR aid",
    ]
    print("\n--- Simple / Complex Boolean Queries ---")
    for q in boolean_queries:
        tokens = tokenize_boolean_query(q)
        postfix = infix_to_postfix(tokens)
        result = eval_postfix(postfix, inverted, all_docs, norm_fn)
        print_result(q, result)

    # Complex Boolean
    complex_queries = [
        "biggest AND ( near OR box )",
        "box AND ( united OR year )",
        "biggest AND ( plane OR wanted OR hour )",
    ]
    print("\n--- Complex Boolean Queries ---")
    for q in complex_queries:
        tokens = tokenize_boolean_query(q)
        postfix = infix_to_postfix(tokens)
        result = eval_postfix(postfix, inverted, all_docs, norm_fn)
        print_result(q, result)

    # Proximity
    proximity_queries = [
        "after years /1",
        "develop solutions /1",
        "keep out /2",
    ]
    print("\n--- Proximity Queries ---")
    for q in proximity_queries:
        t1_raw, t2_raw, k = parse_proximity_query(q)
        t1 = normalize_query_term(t1_raw, stopwords)
        t2 = normalize_query_term(t2_raw, stopwords)
        result = proximity_search(t1, t2, k, positional)
        print_result(q, result)


if __name__ == "__main__":
    run_all_tests()