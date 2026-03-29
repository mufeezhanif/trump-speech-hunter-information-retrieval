import os
import time
import streamlit as st

from src.io_utils import load_documents
from src.preprocessing import load_stopwords, preprocess_text, tokenize, basic_stem_word, porter_stem_word
from src.index_inverted import build_inverted_index, save_inverted_index, load_inverted_index
from src.index_positional import build_positional_index, save_positional_index, load_positional_index
from src.query_parser import tokenize_boolean_query, infix_to_postfix
from src.retrieval_boolean import get_postings, eval_postfix
from src.retrieval_proximity import parse_proximity_query, proximity_search

DATA_DIR = "data/Trump Speechs"
STOPWORD_PATH = "data/Stopword-List.txt"
INVERTED_PATH = "indexes/inverted_index.json"
POSITIONAL_PATH = "indexes/positional_index.json"
STEM_MODE = "porter"


def normalize_query_term(term, stopwords, stem_mode):
    toks = preprocess_text(term, stopwords, stem_mode=stem_mode)
    return toks[0] if toks else ""


def _get_stem_fn(stem_mode):
    if stem_mode == 'porter':
        return porter_stem_word
    return basic_stem_word


def build_and_save_indexes(stem_mode):
    docs = load_documents(DATA_DIR)
    stopwords = load_stopwords(STOPWORD_PATH)
    doc_tokens = {doc_id: preprocess_text(text, stopwords, stem_mode=stem_mode) for doc_id, text in docs.items()}

    inverted = build_inverted_index(doc_tokens)
    positional = build_positional_index(docs, stopwords, _get_stem_fn(stem_mode))

    os.makedirs("indexes", exist_ok=True)
    save_inverted_index(inverted, INVERTED_PATH)
    save_positional_index(positional, POSITIONAL_PATH)

    return inverted, positional, docs, stopwords


def load_indexes():
    inverted = load_inverted_index(INVERTED_PATH)
    positional = load_positional_index(POSITIONAL_PATH)
    docs = load_documents(DATA_DIR)
    stopwords = load_stopwords(STOPWORD_PATH)
    return inverted, positional, docs, stopwords


def ensure_indexes(force_rebuild=False):
    if force_rebuild or not os.path.exists(INVERTED_PATH) or not os.path.exists(POSITIONAL_PATH):
        return build_and_save_indexes(STEM_MODE)
    return load_indexes()


def search_simple_boolean(query, inverted_index, all_docs, stopwords):
    tokens = tokenize_boolean_query(query)
    postfix = infix_to_postfix(tokens)
    norm_fn = lambda t: normalize_query_term(t, stopwords, STEM_MODE)
    return eval_postfix(postfix, inverted_index, all_docs, norm_fn), tokens, postfix


def search_complex_boolean(query, inverted_index, all_docs, stopwords):
    tokens = tokenize_boolean_query(query)
    postfix = infix_to_postfix(tokens)
    norm_fn = lambda t: normalize_query_term(t, stopwords, STEM_MODE)
    return eval_postfix(postfix, inverted_index, all_docs, norm_fn), tokens, postfix


def search_proximity(query, positional_index, stopwords):
    t1_raw, t2_raw, k = parse_proximity_query(query)
    t1 = normalize_query_term(t1_raw, stopwords, STEM_MODE)
    t2 = normalize_query_term(t2_raw, stopwords, STEM_MODE)
    results = proximity_search(t1, t2, k, positional_index)
    return results, t1, t2, k


def main():
    st.set_page_config(page_title="Boolean IR System", layout="wide")
    st.title("Boolean Information Retrieval on Trump Speeches")

    # Sidebar
    with st.sidebar:
        st.header("Index Controls")
        rebuild = st.button("Rebuild Indexes")
        st.markdown("---")
        st.markdown("### Preprocessing Pipeline")
        st.write("- Case Folding (lowercase)")
        st.write("- Stopword Removal")
        st.write(f"- Stemming: `{STEM_MODE}`")
        st.markdown("---")
        st.markdown("### Query Format Examples")
        st.code("Simple:  actions AND wanted", language=None)
        st.code("Simple:  united OR plane", language=None)
        st.code("Simple:  NOT hammer", language=None)
        st.code("Complex: biggest AND ( near OR box )", language=None)
        st.code("Complex: NOT ( united AND plane )", language=None)
        st.code("Proximity: after years /1", language=None)

    # Load or rebuild indexes
    with st.spinner("Loading indexes..."):
        inverted, positional, docs, stopwords = ensure_indexes(force_rebuild=rebuild)
    all_docs = set(docs.keys())

    st.success(f"Indexes loaded: {len(inverted)} terms, {len(docs)} documents")

    # Query input
    col1, col2 = st.columns([1, 3])
    with col1:
        query_type = st.selectbox(
            "Query Type",
            ["Simple Boolean", "Complex Boolean", "Proximity"],
        )
    with col2:
        defaults = {
            "Simple Boolean": "actions AND wanted",
            "Complex Boolean": "biggest AND ( near OR box )",
            "Proximity": "after years /1",
        }
        query_text = st.text_input("Enter Query", value=defaults[query_type])

    if st.button("Search", type="primary"):
        if not query_text.strip():
            st.error("Please enter a query.")
            return

        t0 = time.perf_counter()
        try:
            if query_type == "Simple Boolean":
                results, tokens, postfix = search_simple_boolean(query_text, inverted, all_docs, stopwords)
            elif query_type == "Complex Boolean":
                results, tokens, postfix = search_complex_boolean(query_text, inverted, all_docs, stopwords)
            else:
                results, t1, t2, k = search_proximity(query_text, positional, stopwords)
                tokens = [t1, t2, f"/{k}"]
                postfix = None

            elapsed = (time.perf_counter() - t0) * 1000

            # Results header
            st.subheader(f"Results — {len(results)} document(s) matched")
            st.caption(f"Execution time: {elapsed:.2f} ms")

            # Explainability
            with st.expander("Query Analysis", expanded=False):
                st.write("**Parsed tokens:**", tokens)
                if postfix:
                    st.write("**Postfix expression:**", postfix)
                if query_type == "Proximity":
                    st.write(f"**Normalized terms:** `{t1}`, `{t2}` within **{k}** words")

            # Display results
            if results:
                sorted_results = sorted(results, key=lambda x: int(x))
                st.write("**Matched Document IDs:**", ", ".join(f"speech_{did}" for did in sorted_results))

                st.markdown("---")
                for did in sorted_results:
                    with st.expander(f"speech_{did}.txt"):
                        text = docs.get(did, "")
                        preview = text[:500] + ("..." if len(text) > 500 else "")
                        st.text(preview)
            else:
                st.info("No documents matched the query.")

        except ValueError as e:
            st.error(f"Query error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()