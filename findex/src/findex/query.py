import re
from dataclasses import dataclass
from typing import Set
from findex.index import InvertedIndex

class QueryNode:
    """Базовий клас для всіх вузлів запиту."""
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        raise NotImplementedError
    
    # Перевантаження операторів, щоб можна було писати: Term("a") & (Term("b") | Term("c"))
    def __and__(self, other):
        return And(self, other)
        
    def __or__(self, other):
        return Or(self, other)
        
    def __invert__(self):
        return Not(self)

@dataclass
class Term(QueryNode):
    word: str
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        # Повертає множину ID документів, де є це слово
        postings = index.get(self.word, [])
        return {p.doc_id for p in postings}

@dataclass
class Phrase(QueryNode):
    words: list[str]
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        if not self.words: 
            return set()
        if not index.store_positions:
            raise ValueError("Індекс повинен зберігати позиції (store_positions=True) для фразового пошуку")
        
        # Спочатку знаходимо документи, де є всі слова з фрази
        doc_ids = set.intersection(*[Term(w).evaluate(index) for w in self.words])
        result = set()
        
        for doc_id in doc_ids:
            # Збираємо позиції кожного слова в цьому документі
            word_positions = []
            for w in self.words:
                for p in index[w]:
                    if p.doc_id == doc_id:
                        # p.positions - це кортеж, перетворюємо його для зручності
                        word_positions.append(p.positions)
                        break
            
            # Перевіряємо, чи йдуть слова підряд
            if not word_positions or not word_positions[0]:
                continue
                
            for pos0 in word_positions[0]:
                is_phrase = True
                for i in range(1, len(self.words)):
                    if (pos0 + i) not in word_positions[i]:
                        is_phrase = False
                        break
                if is_phrase:
                    result.add(doc_id)
                    break
                    
        return result

@dataclass
class And(QueryNode):
    left: QueryNode
    right: QueryNode
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        # Перетин множин
        return self.left.evaluate(index) & self.right.evaluate(index)

@dataclass
class Or(QueryNode):
    left: QueryNode
    right: QueryNode
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        # Об'єднання множин
        return self.left.evaluate(index) | self.right.evaluate(index)

@dataclass
class Not(QueryNode):
    child: QueryNode
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        # Усі документи мінус ті, що знайшов дочірній вузол
        all_docs = set(index.doc_meta.keys())
        return all_docs - self.child.evaluate(index)

# ==========================================
# ПАРСЕР (Рекурсивний спуск)
# ==========================================

def parse_query(q: str) -> QueryNode:
    """Перетворює рядок запиту на дерево вузлів (AST)."""
    # Розбиваємо запит на токени: фрази в лапках, дужки, AND, OR, NOT та звичайні слова
    tokens = re.findall(r'"[^"]*"|\(|\)|AND|OR|NOT|\w+', q)
    
    def parse_expr(tokens):
        node, tokens = parse_term(tokens)
        while tokens and tokens[0] == 'OR':
            right, tokens = parse_term(tokens[1:])
            node = Or(node, right)
        return node, tokens
        
    def parse_term(tokens):
        node, tokens = parse_factor(tokens)
        while tokens and tokens[0] == 'AND':
            right, tokens = parse_factor(tokens[1:])
            node = And(node, right)
        return node, tokens
        
    def parse_factor(tokens):
        if not tokens:
            raise ValueError("Неочікуваний кінець запиту")
        
        t = tokens[0]
        if t == 'NOT':
            child, tokens = parse_factor(tokens[1:])
            return Not(child), tokens
        elif t == '(':
            node, tokens = parse_expr(tokens[1:])
            if not tokens or tokens[0] != ')':
                raise ValueError("Не вистачає закриваючої дужки ')'")
            return node, tokens[1:]
        elif t.startswith('"'):
            # Фразовий пошук
            words = t.strip('"').lower().split()
            return Phrase(words), tokens[1:]
        else:
            # Звичайне слово
            return Term(t.lower()), tokens[1:]
            
    node, remaining = parse_expr(tokens)
    if remaining:
        raise ValueError(f"Неочікувані токени в кінці запиту: {remaining}")
    return node