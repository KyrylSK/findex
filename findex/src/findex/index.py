import json
import pickle
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Generator, Tuple
from collections import defaultdict, Counter
from functools import cached_property
from contextlib import contextmanager
from pathlib import Path

@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int

@dataclass(frozen=True)
class Posting:
    doc_id: int
    frequency: int
    positions: Optional[tuple[int, ...]] = None

class InvertedIndex(Mapping[str, List[Posting]]):
    def __init__(self, store_positions: bool = False) -> None:
        self.store_positions: bool = store_positions
        self.postings: defaultdict[str, List[Posting]] = defaultdict(list)
        self.doc_meta: Dict[int, DocMeta] = {}

    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        self.doc_meta[doc_id] = DocMeta(length=len(tokens))
        term_counts: Counter[str] = Counter(tokens)
        term_positions: defaultdict[str, List[int]] = defaultdict(list)
        
        if self.store_positions:
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

        for term, count in term_counts.items():
            positions: Optional[tuple[int, ...]] = tuple(term_positions[term]) if self.store_positions else None
            posting = Posting(doc_id=doc_id, frequency=count, positions=positions)
            self.postings[term].append(posting)

    def merge(self, other: 'InvertedIndex') -> None:
        """Зливає інший індекс з поточним. Оновлює метадані та додає постінги."""
        self.doc_meta.update(other.doc_meta)
        for term, posts in other.postings.items():
            self.postings[term].extend(posts)

    def build_from_pipeline(self, pipeline: Any) -> None:
        for doc_id, tokens in pipeline:
            tokens_list: List[str] = list(tokens)
            self.add_document(doc_id, tokens_list)

    def __getitem__(self, term: str) -> List[Posting]:
        return self.postings.get(term, [])

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __repr__(self) -> str:
        return f"<InvertedIndex: {len(self)} terms, {self.num_docs()} docs>"

    def num_docs(self) -> int:
        return len(self.doc_meta)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, []))

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_meta:
            return 0.0
        total_length: int = sum(meta.length for meta in self.doc_meta.values())
        return float(total_length) / float(self.num_docs())

    def save_pickle(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                'store_positions': self.store_positions,
                'postings': dict(self.postings),
                'doc_meta': self.doc_meta
            }, f)

    @classmethod
    def load_pickle(cls, filepath: str) -> 'InvertedIndex':
        with open(filepath, 'rb') as f:
            data: Dict[str, Any] = pickle.load(f)
            
        index = cls(store_positions=data.get('store_positions', False))
        index.postings = defaultdict(list, data['postings'])
        index.doc_meta = data['doc_meta']
        return index

    def save_json(self, filepath: str) -> None:
        data: Dict[str, Any] = {
            'store_positions': self.store_positions,
            'doc_meta': {str(doc_id): meta.length for doc_id, meta in self.doc_meta.items()},
            'postings': {
                term: [{'doc_id': p.doc_id, 'frequency': p.frequency} for p in posts]
                for term, posts in self.postings.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> 'InvertedIndex':
        with open(filepath, 'r', encoding='utf-8') as f:
            data: Dict[str, Any] = json.load(f)
            
        index = cls(store_positions=data.get('store_positions', False))
        index.doc_meta = {int(doc_id): DocMeta(length=length) for doc_id, length in data['doc_meta'].items()}
        
        for term, posts in data['postings'].items():
            index.postings[term] = [Posting(doc_id=p['doc_id'], frequency=p['frequency']) for p in posts]
            
        return index

@contextmanager
def open_index(filepath: str) -> Generator[InvertedIndex, None, None]:
    is_json: bool = str(filepath).endswith('.json')
    mode: str = 'r' if is_json else 'rb'
    encoding: Optional[str] = 'utf-8' if is_json else None
    
    f = open(filepath, mode, encoding=encoding) # type: ignore
    try:
        if is_json:
            data: Dict[str, Any] = json.load(f)
            index = InvertedIndex(store_positions=data.get('store_positions', False))
            index.doc_meta = {int(doc_id): DocMeta(length=length) for doc_id, length in data['doc_meta'].items()}
            for term, posts in data['postings'].items():
                index.postings[term] = [Posting(doc_id=p['doc_id'], frequency=p['frequency']) for p in posts]
        else:
            data = pickle.load(f)
            index = InvertedIndex(store_positions=data.get('store_positions', False))
            index.postings = defaultdict(list, data['postings'])
            index.doc_meta = data['doc_meta']
            
        yield index
    finally:
        f.close()


def build_partial(docs_chunk: List[Tuple[int, Path]], store_positions: bool) -> InvertedIndex:
    """
    Функція рівня модуля для побудови частини індексу.
    docs_chunk - список кортежів (doc_id, шлях_до_файлу).
    """
    from findex.core import tokenize  # Локальний імпорт для уникнення циклічної залежності
    
    idx = InvertedIndex(store_positions=store_positions)
    for doc_id, file_path in docs_chunk:
        try:
            text = file_path.read_text(encoding="utf-8")
            tokens = tokenize(text)
            idx.add_document(doc_id, tokens)
        except Exception:
            # Ігноруємо помилки читання тут
            pass
    return idx