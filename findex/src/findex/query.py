import re
from dataclasses import dataclass
from typing import Set, Tuple, List
from findex.index import InvertedIndex

class QueryNode:
    """Базовий клас для всіх вузлів запиту."""
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        raise NotImplementedError
    
    def __and__(self, other: 'QueryNode') -> 'QueryNode':
        return And(self, other)
        
    def __or__(self, other: 'QueryNode') -> 'QueryNode':
        return Or(self, other)
        
    def __invert__(self) -> 'QueryNode':
        return Not(self)

@dataclass
class Term(QueryNode):
    word: str
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        postings = index.postings.get(self.word, [])
        return {p.doc_id for p in postings}

@dataclass
class Phrase(QueryNode):
    words: List[str]
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        if not self.words: 
            return set()
        if not index.store_positions:
            raise ValueError("Індекс повинен зберігати позиції")
        
        # Вирішення проблеми з типами: явно вказуємо типи списку та використовуємо методи об'єктів
        term_sets: List[Set[int]] = [Term(w).evaluate(index) for w in self.words]
        doc_ids: Set[int] = term_sets[0].intersection(*term_sets[1:])
        
        result: Set[int] = set()
        
        for doc_id in doc_ids:
            word_positions: List[Tuple[int, ...]] = []
            for w in self.words:
                for p in index.postings.get(w, []):
                    if p.doc_id == doc_id and p.positions is not None:
                        word_positions.append(p.positions)
                        break
            
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
        return self.left.evaluate(index) & self.right.evaluate(index)

@dataclass
class Or(QueryNode):
    left: QueryNode
    right: QueryNode
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

@dataclass
class Not(QueryNode):
    child: QueryNode
    
    def evaluate(self, index: InvertedIndex) -> Set[int]:
        all_docs: Set[int] = set(index.doc_meta.keys())
        return all_docs - self.child.evaluate(index)

def parse_query(q: str) -> QueryNode:
    tokens: List[str] = re.findall(r'"[^"]*"|\(|\)|AND|OR|NOT|\w+', q)
    
    def parse_expr(tokens: List[str]) -> Tuple[QueryNode, List[str]]:
        node, tokens = parse_term(tokens)
        while tokens and tokens[0] == 'OR':
            right, tokens = parse_term(tokens[1:])
            node = Or(node, right)
        return node, tokens
        
    def parse_term(tokens: List[str]) -> Tuple[QueryNode, List[str]]:
        node, tokens = parse_factor(tokens)
        while tokens and tokens[0] == 'AND':
            right, tokens = parse_factor(tokens[1:])
            node = And(node, right)
        return node, tokens
        
    def parse_factor(tokens: List[str]) -> Tuple[QueryNode, List[str]]:
        if not tokens:
            raise ValueError("Неочікуваний кінець запиту")
        
        t = tokens[0]
        if t == 'NOT':
            child, tokens = parse_factor(tokens[1:])
            return Not(child), tokens
        elif t == '(':
            node, tokens = parse_expr(tokens[1:])
            if not tokens or tokens[0] != ')':
                raise ValueError("Не вистачає закриваючої дужки")
            return node, tokens[1:]
        elif t.startswith('"'):
            words = t.strip('"').lower().split()
            return Phrase(words), tokens[1:]
        else:
            return Term(t.lower()), tokens[1:]
            
    node, remaining = parse_expr(tokens)
    if remaining:
        raise ValueError(f"Неочікувані токени: {remaining}")
    return node