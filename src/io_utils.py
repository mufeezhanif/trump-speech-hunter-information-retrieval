import os
from typing import Dict


def load_documents(folder_path: str) -> Dict[str, str]:
    docs = {}
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".txt"):
            continue
        doc_id = fname.replace("speech_", "").replace(".txt", "")
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs