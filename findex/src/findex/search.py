import time
import heapq
from functools import wraps, lru_cache
from typing import List, Tuple
from findex.index import InvertedIndex
from findex.query import parse_query, Term, Phrase
from findex.scoring import Scorer, BM25, SearchResult

# ==========================================
# ДЕКОРАТОРЫ
# ==========================================

def timed(func):
    """Декоратор для замера времени выполнения функции."""
    @wraps(func) # Зберігає оригінальне ім'я та docstring функції
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMED] {func.__name__} виконана за {end - start:.4f} сек")
        return result
    return wrapper

# ==========================================
# СНІПЕТИ
# ==========================================

def build_snippet(text: str, query_words: set[str], window: int = 80) -> str:
    """
    Генерує сніпет: знаходить перше входження будь-якого шуканого слова,
    вирізає вікно тексту навколо нього і підсвічує слово.
    """
    text_lower = text.lower()
    
    # Шукаємо позицію першого слова із запиту
    best_pos = -1
    matched_word = ""
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            matched_word = word
            
    if best_pos == -1:
        # Якщо слова не знайдено (наприклад, через складний NOT запит), беремо початок
        return text[:window*2] + "..."
        
    # Визначаємо межі вікна
    start_pos = max(0, best_pos - window)
    end_pos = min(len(text), best_pos + len(matched_word) + window)
    
    snippet = text[start_pos:end_pos]
    
    # Підсвічуємо знайдене слово (робимо жирним у Markdown/терміналі)
    # Шукаємо його ще раз у вирізаному шматку з урахуванням регістру
    import re
    # Використовуємо регулярку, щоб замінити незалежно від регістру
    pattern = re.compile(re.escape(matched_word), re.IGNORECASE)
    highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)
    
    prefix = "..." if start_pos > 0 else ""
    suffix = "..." if end_pos < len(text) else ""
    return prefix + highlighted + suffix

# ==========================================
# ПОШУК З КЕШУВАННЯМ І РАНЖУВАННЯМ
# ==========================================

# Кешуємо результат: якщо запит повторюється, повертаємо збережений список ID
@lru_cache(maxsize=128)
def _get_matching_docs(query_str: str, index: InvertedIndex) -> frozenset[int]:
    """Парсить запит і знаходить ID відповідних документів. Результат кешується."""
    print(f"[CACHE MISS] Обчислюємо AST для запиту: '{query_str}'")
    ast = parse_query(query_str)
    return frozenset(ast.evaluate(index))

def extract_terms_from_ast(node) -> set[str]:
    """Рекурсивно витягує всі слова із запиту (для генерації сніпетів)."""
    words = set()
    if isinstance(node, Term):
        words.add(node.word)
    elif isinstance(node, Phrase):
        words.update(node.words)
    elif hasattr(node, 'left') and hasattr(node, 'right'): # And, Or
        words.update(extract_terms_from_ast(node.left))
        words.update(extract_terms_from_ast(node.right))
    elif hasattr(node, 'child'): # Not
        words.update(extract_terms_from_ast(node.child))
    return words

@timed
def execute_search(
    query_str: str, 
    index: InvertedIndex, 
    scorer: Scorer = BM25(), 
    top_k: int = 5
) -> List[SearchResult]:
    """
    Головна функція пошуку:
    1. Отримує документи (можливо з кешу).
    2. Оцінює їх через Scorer.
    3. Повертає топ-K результатів.
    """
    # 1. Знаходимо всі документи, що відповідають логічному запиту
    # Оскільки index - об'єкт, ми не можемо передавати його в lru_cache прямо
    # (cache вимагає hashable аргументів), тому перехоплюємо AST окремо
    ast = parse_query(query_str)
    matching_docs = ast.evaluate(index)
    
    if not matching_docs:
        return []

    # Витягуємо всі слова із запиту, щоб знати, що оцінювати
    query_words = extract_terms_from_ast(ast)
    if not query_words:
        return []

    # 2. Оцінюємо кожен документ
    results = []
    for doc_id in matching_docs:
        total_score = 0.0
        # Рахуємо загальний бал документа за всіма словами із запиту
        for word in query_words:
            # Шукаємо posting для цього документа і слова
            postings = index.get(word, [])
            for p in postings:
                if p.doc_id == doc_id:
                    total_score += scorer.score(p, index)
                    break
        
        if total_score > 0:
            results.append(SearchResult(score=total_score, doc_id=doc_id))

    # 3. Вибираємо ТОП-K документів за найвищим балом
    # heapq.nlargest автоматично використовує поле sort_score (яке ми зробили від'ємним)
    top_results = heapq.nlargest(top_k, results)
    
    return top_results