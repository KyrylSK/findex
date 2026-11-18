import math
from typing import Protocol
from dataclasses import dataclass, field
import numpy as np
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
    """Протокол (інтерфейс) для всіх алгоритмів ранжування."""
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        ...

    def score_arrays(
        self, doc_ids: np.ndarray, freqs: np.ndarray, df: int, index: InvertedIndex
    ) -> np.ndarray:
        ...


class TfIdf:
    """Векторизований алгоритм ранжування TF-IDF."""
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        tf = float(posting.frequency)
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
        idf = math.log(N / float(df))
        return tf * idf

    def score_arrays(
        self, doc_ids: np.ndarray, freqs: np.ndarray, df: int, index: InvertedIndex
    ) -> np.ndarray:
        if df == 0 or len(doc_ids) == 0:
            return np.zeros(len(doc_ids), dtype=np.float64)
        N = float(index.num_docs())
        idf = np.log(N / float(df))
        return freqs.astype(np.float64) * idf


class BM25:
    """Векторизований алгоритм ранжування BM25."""
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

        doc_length = float(index.doc_lengths[posting.doc_id])
        avgdl = index.avg_doc_length

        idf = math.log((N - float(df) + 0.5) / (float(df) + 0.5) + 1.0)

        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / avgdl))

        return idf * (numerator / denominator)

    def score_arrays(
        self, doc_ids: np.ndarray, freqs: np.ndarray, df: int, index: InvertedIndex
    ) -> np.ndarray:
        if df == 0 or len(doc_ids) == 0:
            return np.zeros(len(doc_ids), dtype=np.float64)
        N = float(index.num_docs())
        avgdl = index.avg_doc_length
        if avgdl == 0.0:
            return np.zeros(len(doc_ids), dtype=np.float64)

        tf = freqs.astype(np.float64)
        doc_lens = index.doc_lengths[doc_ids].astype(np.float64)

        idf = np.log((N - float(df) + 0.5) / (float(df) + 0.5) + 1.0)
        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_lens / avgdl))
        return idf * (numerator / denominator)