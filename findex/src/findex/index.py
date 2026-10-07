from dataclasses import dataclass
from typing import Optional
from collections import defaultdict, Counter

import pickle
import json

@dataclass(frozen=True, slots=True)
class DocMeta:
    """Метадані документа (наприклад, кількість слів для ранжування)."""
    length: int

@dataclass(frozen=True)
class Posting:
    """Постінг: id документа, частота терміна і опціонально позиції."""
    doc_id: int
    frequency: int
    # Позиції токенів — опційно (кортеж займає менше пам'яті, ніж список)
    positions: Optional[tuple[int, ...]] = None

class InvertedIndex:
    def __init__(self, store_positions: bool = False):
        self.store_positions = store_positions
        # defaultdict автоматично створює порожній список, якщо терміна ще немає в словнику
        self.postings: defaultdict[str, list[Posting]] = defaultdict(list)
        # Словник для зберігання метаданих документів (довжини)
        self.doc_meta: dict[int, DocMeta] = {}

    def add_document(self, doc_id: int, tokens: list[str]):
        """Додає розпарсений документ (список токенів) до індексу."""
        # Зберігаємо довжину документа
        self.doc_meta[doc_id] = DocMeta(length=len(tokens))

        # Counter швидко рахує, скільки разів кожне слово зустрілося в документі
        term_counts = Counter(tokens)
        
        # Якщо увімкнено збереження позицій
        term_positions = defaultdict(list)
        if self.store_positions:
            for pos, token in enumerate(tokens):
                term_positions[token].append(pos)

        # Створюємо постінги для кожного унікального слова
        for term, count in term_counts.items():
            positions = tuple(term_positions[term]) if self.store_positions else None
            posting = Posting(doc_id=doc_id, frequency=count, positions=positions)
            self.postings[term].append(posting)
            
    def build_from_pipeline(self, pipeline):
        """Будує індекс одним проходом по лінивому пайплайну з першої лаби."""
        for doc_id, tokens in pipeline:
            # Оскільки пайплайн лінивий, ми перетворюємо генератор токенів на список один раз
            tokens_list = list(tokens) 
            self.add_document(doc_id, tokens_list)
    def _merge_and(self, p1: list[Posting], p2: list[Posting]) -> list[Posting]:
        """Перетин (AND) двох відсортованих списків постінгів через два вказівники."""
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i].doc_id == p2[j].doc_id:
                # Документ містить обидва слова. Для простоти беремо постінг з першого списку.
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i].doc_id < p2[j].doc_id:
                i += 1
            else:
                j += 1
        return result

    def _set_and(self, p1: list[Posting], p2: list[Posting]) -> list[Posting]:
        """Перетин (AND) двох списків постінгів через вбудовані множини Python (set)."""
        # Створюємо множину id документів другого списку для швидкого пошуку
        ids2 = {p.doc_id for p in p2}
        # Залишаємо тільки ті постінги з першого, які є в другому
        return [p for p in p1 if p.doc_id in ids2]

    def search(self, query_terms: list[str], engine: str = "merge") -> list[Posting]:
        """
        Пошук документів, що містять УСІ терміни (логічне AND).
        engine може бути 'merge' або 'set'.
        """
        if not query_terms:
            return []

        # Дістаємо списки постінгів для кожного слова
        lists = [self.postings.get(term, []) for term in query_terms]

        # Якщо хоча б одного слова немає в індексі, то перетин буде порожнім
        if any(not lst for lst in lists):
            return []

        # Оптимізація: починаємо перетин з найкоротших списків (менше роботи)
        lists.sort(key=len)

        result = lists[0]
        for i in range(1, len(lists)):
            if engine == "merge":
                result = self._merge_and(result, lists[i])
            elif engine == "set":
                result = self._set_and(result, lists[i])
            else:
                raise ValueError(f"Невідомий рушій пошуку: {engine}")
                
        return result
    def save_pickle(self, filepath: str):
        """Зберігає індекс у бінарний файл за допомогою pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'store_positions': self.store_positions,
                'postings': self.postings,
                'doc_meta': self.doc_meta
            }, f)

    @classmethod
    def load_pickle(cls, filepath: str):
        """
        Завантажує індекс з pickle.
        УВАГА ДЛЯ ЗАХИСТУ: pickle НЕБЕЗПЕЧНИЙ для завантаження файлів з невідомих джерел!
        Причина: pickle може зберігати не тільки дані, а й інструкції для створення об'єктів.
        Зловмисник може підкласти файл, який при розпакуванні (unpickling) виконає
        довільний системний код (наприклад, видалить файли або завантажить вірус) - 
        це називається RCE (Remote Code Execution).
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        index = cls(store_positions=data.get('store_positions', False))
        index.postings = data['postings']
        index.doc_meta = data['doc_meta']
        return index

    def save_json(self, filepath: str):
        """Другий формат: зберігає індекс у JSON."""
        # JSON не вміє сам зберігати dataclass, тому перетворюємо їх у словники
        data = {
            'store_positions': self.store_positions,
            'doc_meta': {doc_id: meta.length for doc_id, meta in self.doc_meta.items()},
            'postings': {
                term: [{'doc_id': p.doc_id, 'frequency': p.frequency} for p in posts]
                for term, posts in self.postings.items()
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, filepath: str):
        """Завантажує індекс з JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        index = cls(store_positions=data.get('store_positions', False))
        
        # Відновлюємо DocMeta
        index.doc_meta = {int(doc_id): DocMeta(length=length) for doc_id, length in data['doc_meta'].items()}
        
        # Відновлюємо Posting
        for term, posts in data['postings'].items():
            index.postings[term] = [Posting(doc_id=p['doc_id'], frequency=p['frequency']) for p in posts]
            
        return index
if __name__ == '__main__':
    import argparse
    import time
    import tracemalloc
    from pathlib import Path
    # Імпортуємо твої реальні функції з core.py
    from findex.core import iter_documents, tokenize

    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу.")
    parser.add_argument("data_dir", help="Шлях до директорії з документами")
    parser.add_argument("--out", required=True, help="Шлях для збереження індексу (.bin або .json)")
    parser.add_argument("--positions", action="store_true", help="Зберігати позиції токенів")
    args = parser.parse_args()

    print(f"Будуємо індекс з {args.data_dir}...")
    
    tracemalloc.start()
    start_time = time.time()

    idx = InvertedIndex(store_positions=args.positions)
    
    # Збираємо пайплайн: читаємо файли і одразу токенізуємо їх
    def create_pipeline(path):
        for doc_id, text in enumerate(iter_documents(Path(path))):
            yield doc_id, tokenize(text)

    pipeline = create_pipeline(args.data_dir)
    idx.build_from_pipeline(pipeline)

    build_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Індекс побудовано за {build_time:.2f} сек.")
    print(f"Пікове використання пам'яті: {peak / 1024 / 1024:.2f} MB")

    print(f"Зберігаємо у {args.out}...")
    if args.out.endswith('.json'):
        idx.save_json(args.out)
    else:
        idx.save_pickle(args.out)
    print("Готово!")