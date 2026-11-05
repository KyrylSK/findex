import time
import heapq
import re
from functools import wraps, lru_cache
from typing import List, Set, FrozenSet, Callable, TypeVar, ParamSpec

from findex.index import InvertedIndex, Posting
from findex.query import parse_query, Term, Phrase, QueryNode, And, Or, Not
from findex.scoring import Scorer, SearchResult

P = ParamSpec('P')
R = TypeVar('R')

def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для замера времени выполнения функции."""
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
        return text[:window*2] + "..."
        
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
    print(f"[CACHE MISS] Обчислюємо AST для запиту: '{query_str}'")
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

@timed
def execute_search(
    query_str: str, 
    index: InvertedIndex, 
    scorer: Scorer,
    top_k: int = 5
) -> List[SearchResult]:
    ast = parse_query(query_str)
    matching_docs: Set[int] = ast.evaluate(index)
    
    if not matching_docs:
        return []

    query_words: Set[str] = extract_terms_from_ast(ast)
    if not query_words:
        return []

    results: List[SearchResult] = []
    for doc_id in matching_docs:
        total_score: float = 0.0
        for word in query_words:
            postings: List[Posting] = index.postings.get(word, [])
            for p in postings:
                if p.doc_id == doc_id:
                    total_score += scorer.score(p, index)
                    break
        
        if total_score > 0:
            results.append(SearchResult(score=total_score, doc_id=doc_id))

    top_results: List[SearchResult] = heapq.nlargest(top_k, results)
    return top_results