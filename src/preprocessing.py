from typing import List, Set
from nltk.stem import PorterStemmer

_porter = PorterStemmer()


def porter_stem_word(word: str) -> str:
    return _porter.stem(word)


def load_stopwords(path: str) -> Set[str]:
    stopwords = set()
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords

def tokenize(text:str)-> List[str]:
    text = text.lower()
    tokens = []
    current = []
    
    for ch in text:
        if ("a" <=ch <="z") or ("0"<=ch <="9"):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
        
    return tokens

def basic_stem_word(word: str)-> str:
    if len(word)<=3:
        return word
    
    rules = [
        ("ingly", ""),
        ("edly", ""),
        ("ing", ""),
        ("ed", ""),
        ("ies", "y"),
        ("sses", "ss"),
        ("s", ""),
        ("ly", ""),
        ("ment", ""),
    ]
    
    for suffix, replacement in rules:
        if word.endswith(suffix)  and len(word) > len(suffix) +2:
            return word[: -len(suffix)] + replacement
        
    return word

def normalize_tokens(tokens: List[str], stopwords: Set[str], stem_mode: str = "porter")->List[str]:
    cleaned = []
    
    if stem_mode == 'porter':
        stem_fn = porter_stem_word
    else:
        stem_fn = basic_stem_word
        
    for tok in tokens:
        if tok in stopwords:
            continue
        cleaned.append(stem_fn(tok))
    return cleaned

def preprocess_text(text: str, stopwords: Set[str], stem_mode:str='porter') -> List[str]:
    tokens = tokenize(text)
    return normalize_tokens(tokens, stopwords, stem_mode=stem_mode)