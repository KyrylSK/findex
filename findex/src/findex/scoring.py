import math
from typing import Protocol
from dataclasses import dataclass, field
from findex.index import InvertedIndex, Posting

@dataclass(order=True)
class SearchResult:
    """
    Класс для хранения результата поиска.
    order=True позволяет автоматически сортировать объекты по полю score 
    (чем больше score, тем выше документ).
    """
    # Сортировка будет идти по этому полю. 
    # Ставим знак минус перед баллом, чтобы сортировалось по убыванию (от большего к меньшему)
    sort_score: float = field(init=False, repr=False)
    
    score: float
    doc_id: int
    
    def __post_init__(self):
        self.sort_score = -self.score

class Scorer(Protocol):
    """
    Протокол (интерфейс) для всех алгоритмов ранжирования.
    Любой класс-алгоритм должен реализовать метод score().
    """
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        ...

class TfIdf:
    """Алгоритм ранжирования TF-IDF."""
    def score(self, posting: Posting, index: InvertedIndex) -> float:
        # TF (Term Frequency) - частота слова в конкретном документе
        tf = posting.frequency
        
        # DF (Document Frequency) - в скольких документах вообще есть это слово
        # Ищем слово в индексе (у нас index уже ведет себя как словарь)
        df = 0
        for term, postings_list in index.items():
            for p in postings_list:
                if p.doc_id == posting.doc_id and p.frequency == posting.frequency:
                    df = index.df(term)
                    break
            if df > 0:
                break
                
        # Защита от деления на ноль, если слово не найдено
        if df == 0:
            return 0.0
            
        # IDF (Inverse Document Frequency)
        N = index.num_docs()
        idf = math.log(N / df)
        
        return tf * idf

class BM25:
    """
    Алгоритм ранжирования BM25.
    k1 контролирует "насыщение" частоты (обычно 1.2 - 2.0).
    b контролирует влияние длины документа (обычно 0.75).
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, posting: Posting, index: InvertedIndex) -> float:
        # Находим DF для этого термина (аналогично TfIdf)
        df = 0
        for term, postings_list in index.items():
            for p in postings_list:
                if p.doc_id == posting.doc_id and p.frequency == posting.frequency:
                    df = index.df(term)
                    break
            if df > 0:
                break

        if df == 0:
            return 0.0

        N = index.num_docs()
        tf = posting.frequency
        
        # Получаем длину конкретного документа из doc_meta
        doc_length = index.doc_meta[posting.doc_id].length
        avgdl = index.avg_doc_length

        # Формула IDF для BM25 (немного отличается от стандартной)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        
        # Формула TF для BM25 с учетом длины документа
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avgdl))
        
        return idf * (numerator / denominator)