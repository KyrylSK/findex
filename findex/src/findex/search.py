import time
import math
import re
from functools import wraps, lru_cache
from typing import List, Set, FrozenSet, Callable, TypeVar, ParamSpec
import numpy as np

from findex.index import InvertedIndex, Posting
from findex.query import parse_query, Term, Phrase, QueryNode, And, Or, Not
from findex.scoring import Scorer, SearchResult, BM25

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для вимірювання часу виконання функції."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMED] {func.__name__} виконана за {end - start:.4f} сек")
        return result
    return wrapper


def build_snippet(text: str, query_words: Set[str], window: int = 80) -> str:
    text_lower = text.lower()
    best_pos = -1
    matched_word = ""
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            matched_word = word

    if best_pos == -1:
        return text[: window * 2] + "..."

    start_pos = max(0, best_pos - window)
    end_pos = min(len(text), best_pos + len(matched_word) + window)

    snippet = text[start_pos:end_pos]

    pattern = re.compile(re.escape(matched_word), re.IGNORECASE)
    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)

    prefix = "..." if start_pos > 0 else ""
    suffix = "..." if end_pos < len(text) else ""
    return prefix + highlighted + suffix


@lru_cache(maxsize=128)
def _get_matching_docs(query_str: str, index: InvertedIndex) -> FrozenSet[int]:
    ast = parse_query(query_str)
    return frozenset(ast.evaluate(index))


def extract_terms_from_ast(node: QueryNode) -> Set[str]:
    words: Set[str] = set()
    if isinstance(node, Term):
        words.add(node.word)
    elif isinstance(node, Phrase):
        words.update(node.words)
    elif isinstance(node, (And, Or)):
        words.update(extract_terms_from_ast(node.left))
        words.update(extract_terms_from_ast(node.right))
    elif isinstance(node, Not):
        words.update(extract_terms_from_ast(node.child))
    return words


def execute_search_scalar(
    query_str: str,
    index: InvertedIndex,
    scorer: Scorer,
    top_k: int = 5,
) -> List[SearchResult]:
    """Еталонна скалярна реалізація пошуку з лаби 3 для порівняння та тестів."""
    ast = parse_query(query_str)
    matching_docs: Set[int] = ast.evaluate(index)
    if not matching_docs:
        return []

    query_words: Set[str] = extract_terms_from_ast(ast)
    if not query_words:
        return []

    N = float(index.num_docs())
    avgdl = index.avg_doc_length
    results: List[SearchResult] = []
    for doc_id in matching_docs:
        total_score: float = 0.0
        for word in query_words:
            postings: List[Posting] = index.postings.get(word, [])
            df = len(postings)
            if df == 0:
                continue
            for p in postings:
                if p.doc_id == doc_id:
                    tf = float(p.frequency)
                    if isinstance(scorer, BM25):
                        doc_len = float(index.doc_lengths[doc_id])
                        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                        num = tf * (scorer.k1 + 1.0)
                        den = tf + scorer.k1 * (1.0 - scorer.b + scorer.b * (doc_len / avgdl))
                        total_score += idf * (num / den)
                    else:
                        total_score += tf * math.log(N / float(df))
                    break
        if total_score > 0:
            results.append(SearchResult(score=total_score, doc_id=doc_id))

    return sorted(results)[:top_k]


def execute_search(
    query_str: str,
    index: InvertedIndex,
    scorer: Scorer,
    top_k: int = 5,
) -> List[SearchResult]:
    """Векторизований пошук на NumPy з вибором top-k через np.argpartition."""
    ast = parse_query(query_str)
    matching_docs: Set[int] = ast.evaluate(index)

    if not matching_docs:
        return []

    query_words: Set[str] = extract_terms_from_ast(ast)
    if not query_words:
        return []

    max_doc_id = max(matching_docs)
    scores_vec = np.zeros(max_doc_id + 1, dtype=np.float64)

    for word in query_words:
        doc_ids, freqs = index.get_term_arrays(word)
        if len(doc_ids) == 0:
            continue
        df = len(doc_ids)
        term_scores = scorer.score_arrays(doc_ids, freqs, df, index)
        valid_mask = doc_ids <= max_doc_id
        np.add.at(scores_vec, doc_ids[valid_mask], term_scores[valid_mask])

    cand_ids = np.fromiter(matching_docs, dtype=np.int32, count=len(matching_docs))
    cand_scores = scores_vec[cand_ids]
    pos_mask = cand_scores > 0.0
    cand_ids = cand_ids[pos_mask]
    cand_scores = cand_scores[pos_mask]

    n_cand = len(cand_scores)
    if n_cand == 0 or top_k <= 0:
        return []

    if n_cand <= top_k:
        top_idx = np.lexsort((cand_ids, -cand_scores))
    else:
        part_idx = np.argpartition(-cand_scores, top_k - 1)[:top_k]
        min_top_score = float(np.min(cand_scores[part_idx]))
        tied_mask = cand_scores >= min_top_score
        if int(np.sum(tied_mask)) == top_k:
            top_idx = part_idx[np.lexsort((cand_ids[part_idx], -cand_scores[part_idx]))]
        else:
            tied_indices = np.flatnonzero(tied_mask)
            top_idx = tied_indices[
                np.lexsort((cand_ids[tied_indices], -cand_scores[tied_indices]))[:top_k]
            ]

    return [
        SearchResult(score=float(cand_scores[i]), doc_id=int(cand_ids[i]))
        for i in top_idx
    ]
