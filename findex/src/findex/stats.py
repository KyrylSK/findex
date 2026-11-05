from pathlib import Path
from typing import Counter as CounterType
from collections import Counter
from findex.core import iter_documents, tokenize

def get_corpus_stats(corpus_path: Path) -> tuple[int, int, list[tuple[str, int]]]:
    """Повертає загальну кількість слів, унікальних слів та топ-10 найчастіших."""
    total_words: int = 0
    vocab: CounterType[str] = Counter()
    
    for text in iter_documents(corpus_path):
        tokens = tokenize(text)
        total_words += len(tokens)
        vocab.update(tokens)
        
    unique_words = len(vocab)
    top_10 = vocab.most_common(10)
    
    return total_words, unique_words, top_10