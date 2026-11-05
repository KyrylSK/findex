import math
from typing import Protocol
from dataclasses import dataclass, field
from findex.index import InvertedIndex, Posting

@dataclass(order=True)
class SearchResult:
    """Клас для зберігання результату пошуку."""
    sort_score: float = field(init=False, repr=False)
    score: float
    doc_id: int
    
    def __post_init__(self) -> None:
        self.sort_score = -self.score

class Scorer(Protocol):
    """
    Протокол (інтерфейс) для всіх алгоритмів ранжування.
    """
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        ...

class TfIdf:
    """Алгоритм ранжування TF-IDF."""
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        tf = float(posting.frequency)
        df = 0
        # Оскільки index тепер Mapping, type checker може сваритися на items() без типів.
        # Якщо pyright видасть тут помилку - ми її виправимо пізніше, поки залишаємо логіку як є.
        for term, postings_list in index.postings.items():
            for p in postings_list:
                if p.doc_id == posting.doc_id and p.frequency == posting.frequency:
                    df = index.df(term)
                    break
            if df > 0:
                break
                
        if df == 0:
            return 0.0
            
        N = float(index.num_docs())
        idf = math.log(N / float(df))
        return tf * idf

class BM25:
    """Алгоритм ранжування BM25."""
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, posting: Posting, index: InvertedIndex) -> float:
        df = 0
        for term, postings_list in index.postings.items():
            for p in postings_list:
                if p.doc_id == posting.doc_id and p.frequency == posting.frequency:
                    df = index.df(term)
                    break
            if df > 0:
                break

        if df == 0:
            return 0.0

        N = float(index.num_docs())
        tf = float(posting.frequency)
        
        doc_length = float(index.doc_meta[posting.doc_id].length)
        avgdl = index.avg_doc_length

        idf = math.log((N - float(df) + 0.5) / (float(df) + 0.5) + 1.0)
        
        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / avgdl))
        
        return idf * (numerator / denominator)