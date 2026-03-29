# Boolean IR System — Trump Speeches

Information Retrieval Assignment 1. Implements a Boolean retrieval system over 56 Trump speech documents.

## Approach

**Preprocessing:** Custom tokenizer (alphanumeric-only) → lowercase → stopword removal → NLTK PorterStemmer. Same pipeline applied to both documents and queries.

**Inverted Index:** Maps each stemmed term to the set of document IDs containing it. Supports AND, OR, NOT queries including nested parentheses via shunting-yard postfix evaluation.

**Positional Index:** Maps each stemmed term to its original token positions per document (stopwords skipped, positions kept from raw text). Used for proximity queries.

**Proximity Queries:** Format `term1 term2 /k` — returns documents where the two terms appear with exactly **k words between them**.

## Structure

```
src/
  preprocessing.py   # tokenizer, stopword removal, stemming
  io_utils.py        # document loader
  index_inverted.py  # build/save/load inverted index
  index_positional.py # build/save/load positional index
  query_parser.py    # tokenize query, infix-to-postfix
  retrieval_boolean.py # eval postfix with set operations
  retrieval_proximity.py # proximity search
  evaluate.py        # batch test runner
app.py               # Streamlit UI
indexes/             # saved JSON index files
data/                # speeches + stopword list + query list
```
## UI
<img width="1600" height="789" alt="image" src="https://github.com/user-attachments/assets/152df663-e1e3-423e-88c7-e6e1622a1f86" />

## Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app (indexes are built automatically on first launch)
streamlit run app.py

# 3. Or run evaluation tests against Query List
python -m src.evaluate
```

## Query Formats

| Type | Example |
|------|---------|
| Simple Boolean | `actions AND wanted` |
| NOT | `NOT hammer` |
| Complex Boolean | `biggest AND ( near OR box )` |
| Proximity | `after years /1` |
