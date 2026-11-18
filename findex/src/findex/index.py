import json
import pickle
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Generator, Tuple
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
import numpy as np


@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int


@dataclass(frozen=True)
class Posting:
    doc_id: int
    frequency: int
    positions: Optional[tuple[int, ...]] = None


class _PostingsMapping(Mapping[str, List[Posting]]):
    """Зворотньо-сумісне представлення постінгів поверх масивів NumPy int32."""

    def __init__(self, index: "InvertedIndex") -> None:
        self._index = index

    def __getitem__(self, term: str) -> List[Posting]:
        return self._index[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, term: object) -> bool:
        return term in self._index


class InvertedIndex(Mapping[str, List[Posting]]):
    def __init__(self, store_positions: bool = False) -> None:
        self.store_positions: bool = store_positions
        self._postings_doc_ids: Dict[str, np.ndarray] = {}
        self._postings_freqs: Dict[str, np.ndarray] = {}
        self._postings_positions: Dict[str, List[Optional[tuple[int, ...]]]] = {}
        self._staging_doc_ids: defaultdict[str, List[int]] = defaultdict(list)
        self._staging_freqs: defaultdict[str, List[int]] = defaultdict(list)
        self.doc_lengths: np.ndarray = np.zeros(0, dtype=np.int32)
        self.doc_meta: Dict[int, DocMeta] = {}

    @property
    def postings(self) -> Mapping[str, List[Posting]]:
        return _PostingsMapping(self)

    def _set_doc_length(self, doc_id: int, length: int) -> None:
        if doc_id >= len(self.doc_lengths):
            new_arr = np.zeros(doc_id + 1, dtype=np.int32)
            if len(self.doc_lengths) > 0:
                new_arr[: len(self.doc_lengths)] = self.doc_lengths
            self.doc_lengths = new_arr
        self.doc_lengths[doc_id] = np.int32(length)
        self.doc_meta[doc_id] = DocMeta(length=int(length))

    def _flush_term(self, term: str) -> None:
        if term not in self._staging_doc_ids or not self._staging_doc_ids[term]:
            return
        new_ids = np.asarray(self._staging_doc_ids.pop(term), dtype=np.int32)
        new_freqs = np.asarray(self._staging_freqs.pop(term), dtype=np.int32)
        if term in self._postings_doc_ids:
            self._postings_doc_ids[term] = np.concatenate(
                [self._postings_doc_ids[term], new_ids]
            ).astype(np.int32, copy=False)
            self._postings_freqs[term] = np.concatenate(
                [self._postings_freqs[term], new_freqs]
            ).astype(np.int32, copy=False)
        else:
            self._postings_doc_ids[term] = new_ids
            self._postings_freqs[term] = new_freqs

    def finalize(self) -> None:
        """Конвертує всі буферизовані постінги у неперервні масиви np.int32."""
        for term in list(self._staging_doc_ids.keys()):
            self._flush_term(term)

    def get_term_arrays(self, term: str) -> Tuple[np.ndarray, np.ndarray]:
        """Повертає пару масивів (doc_ids: np.int32, freqs: np.int32) для терму."""
        if term in self._staging_doc_ids:
            self._flush_term(term)
        doc_ids = self._postings_doc_ids.get(term)
        if doc_ids is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        return doc_ids, self._postings_freqs[term]

    def _all_terms(self) -> List[str]:
        if not self._staging_doc_ids:
            return list(self._postings_doc_ids.keys())
        return list(dict.fromkeys([*self._postings_doc_ids.keys(), *self._staging_doc_ids.keys()]))

    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        self._set_doc_length(doc_id, len(tokens))
        term_counts: Counter[str] = Counter(tokens)
        term_positions: defaultdict[str, List[int]] = defaultdict(list)

        if self.store_positions:
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

        for term, count in term_counts.items():
            self._staging_doc_ids[term].append(int(doc_id))
            self._staging_freqs[term].append(int(count))
            if self.store_positions:
                self._postings_positions.setdefault(term, []).append(tuple(term_positions[term]))

    def merge(self, other: "InvertedIndex") -> None:
        """Зливає інший індекс з поточним у масивах NumPy int32."""
        self.finalize()
        other.finalize()
        for doc_id, meta in other.doc_meta.items():
            self._set_doc_length(doc_id, meta.length)

        for term in other._postings_doc_ids:
            other_ids = other._postings_doc_ids[term]
            other_freqs = other._postings_freqs[term]
            if term in self._postings_doc_ids:
                self._postings_doc_ids[term] = np.concatenate(
                    [self._postings_doc_ids[term], other_ids]
                ).astype(np.int32, copy=False)
                self._postings_freqs[term] = np.concatenate(
                    [self._postings_freqs[term], other_freqs]
                ).astype(np.int32, copy=False)
            else:
                self._postings_doc_ids[term] = other_ids.copy()
                self._postings_freqs[term] = other_freqs.copy()
            if self.store_positions and term in other._postings_positions:
                self._postings_positions.setdefault(term, []).extend(other._postings_positions[term])

    def build_from_pipeline(self, pipeline: Any) -> None:
        for doc_id, tokens in pipeline:
            tokens_list: List[str] = list(tokens)
            self.add_document(doc_id, tokens_list)
        self.finalize()

    def __getitem__(self, term: str) -> List[Posting]:
        doc_ids, freqs = self.get_term_arrays(term)
        if len(doc_ids) == 0:
            return []
        positions_list = self._postings_positions.get(term)
        if self.store_positions and positions_list and len(positions_list) == len(doc_ids):
            return [
                Posting(doc_id=int(d), frequency=int(f), positions=pos)
                for d, f, pos in zip(doc_ids, freqs, positions_list)
            ]
        return [Posting(doc_id=int(d), frequency=int(f)) for d, f in zip(doc_ids, freqs)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._all_terms())

    def __len__(self) -> int:
        return len(self._all_terms())

    def __contains__(self, term: object) -> bool:
        return term in self._postings_doc_ids or term in self._staging_doc_ids

    def __repr__(self) -> str:
        return f"<InvertedIndex: {len(self)} terms, {self.num_docs()} docs>"

    def num_docs(self) -> int:
        return len(self.doc_meta)

    def df(self, term: str) -> int:
        doc_ids, _ = self.get_term_arrays(term)
        return int(len(doc_ids))

    @property
    def avg_doc_length(self) -> float:
        n = self.num_docs()
        if n == 0:
            return 0.0
        return float(np.sum(self.doc_lengths)) / float(n)

    def save_pickle(self, filepath: str) -> None:
        self.finalize()
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "store_positions": self.store_positions,
                    "postings_doc_ids": self._postings_doc_ids,
                    "postings_freqs": self._postings_freqs,
                    "postings_positions": self._postings_positions,
                    "doc_lengths": self.doc_lengths,
                    "doc_meta": self.doc_meta,
                },
                f,
            )

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "InvertedIndex":
        index = cls(store_positions=data.get("store_positions", False))
        if "postings_doc_ids" in data and "postings_freqs" in data:
            index._postings_doc_ids = {
                t: np.asarray(arr, dtype=np.int32) for t, arr in data["postings_doc_ids"].items()
            }
            index._postings_freqs = {
                t: np.asarray(arr, dtype=np.int32) for t, arr in data["postings_freqs"].items()
            }
            index._postings_positions = data.get("postings_positions", {})
            index.doc_meta = data.get("doc_meta", {})
            if "doc_lengths" in data:
                index.doc_lengths = np.asarray(data["doc_lengths"], dtype=np.int32)
            else:
                for doc_id, meta in index.doc_meta.items():
                    index._set_doc_length(int(doc_id), meta.length)
        else:
            for doc_id, meta in data.get("doc_meta", {}).items():
                length = meta.length if hasattr(meta, "length") else int(meta)
                index._set_doc_length(int(doc_id), length)
            for term, posts in data.get("postings", {}).items():
                if posts and isinstance(posts[0], Posting):
                    index._postings_doc_ids[term] = np.array([p.doc_id for p in posts], dtype=np.int32)
                    index._postings_freqs[term] = np.array([p.frequency for p in posts], dtype=np.int32)
                elif posts:
                    index._postings_doc_ids[term] = np.array([p["doc_id"] for p in posts], dtype=np.int32)
                    index._postings_freqs[term] = np.array([p["frequency"] for p in posts], dtype=np.int32)
        return index

    @classmethod
    def load_pickle(cls, filepath: str) -> "InvertedIndex":
        with open(filepath, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
        return cls._from_dict(data)

    def save_json(self, filepath: str) -> None:
        self.finalize()
        data: Dict[str, Any] = {
            "store_positions": self.store_positions,
            "doc_meta": {str(doc_id): meta.length for doc_id, meta in self.doc_meta.items()},
            "postings": {
                term: [
                    {"doc_id": int(d), "frequency": int(freq)}
                    for d, freq in zip(self._postings_doc_ids[term], self._postings_freqs[term])
                ]
                for term in self._postings_doc_ids
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "InvertedIndex":
        with open(filepath, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return cls._from_dict(data)


@contextmanager
def open_index(filepath: str) -> Generator[InvertedIndex, None, None]:
    is_json: bool = str(filepath).endswith(".json")
    mode: str = "r" if is_json else "rb"
    encoding: Optional[str] = "utf-8" if is_json else None

    f = open(filepath, mode, encoding=encoding)  # type: ignore
    try:
        if is_json:
            data: Dict[str, Any] = json.load(f)
        else:
            data = pickle.load(f)
        yield InvertedIndex._from_dict(data)
    finally:
        f.close()


def build_partial(docs_chunk: List[Tuple[int, Path]], store_positions: bool) -> InvertedIndex:
    """Функція рівня модуля для побудови частини індексу."""
    from findex.core import tokenize

    idx = InvertedIndex(store_positions=store_positions)
    for doc_id, file_path in docs_chunk:
        try:
            text = file_path.read_text(encoding="utf-8")
            tokens = tokenize(text)
            idx.add_document(doc_id, tokens)
        except Exception:
            pass
    idx.finalize()
    return idx