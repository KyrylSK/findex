import argparse
import time
import tracemalloc
from findex.index import InvertedIndex, Posting, DocMeta
import __main__

# Це хак для pickle, щоб він зміг знайти ці класи під час завантаження
__main__.Posting = Posting
__main__.DocMeta = DocMeta

def main():
    parser = argparse.ArgumentParser(description="Пошук по інвертованому індексу.")
    parser.add_argument("index_file", help="Шлях до файлу індексу (.bin або .json)")
    parser.add_argument("query", help="Пошуковий запит (слова через пробіл)")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", 
                        help="Алгоритм перетину списків (merge або set)")
    args = parser.parse_args()

    print(f"Завантажуємо індекс з {args.index_file}...")
    
    # Вимірюємо пам'ять і час для завантаження (вимога лаби)
    tracemalloc.start()
    start_time = time.time()

    if args.index_file.endswith('.json'):
        idx = InvertedIndex.load_json(args.index_file)
    else:
        idx = InvertedIndex.load_pickle(args.index_file)

    load_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Завантажено за {load_time:.4f} сек.")
    print(f"Пікове використання пам'яті: {peak / 1024 / 1024:.2f} MB")

    # Розбиваємо запит на окремі слова
    query_terms = args.query.lower().split()
    print(f"\nШукаємо: {query_terms} (рушій: {args.engine})")
    
    # Вимірюємо час самого пошуку
    start_time = time.time()
    results = idx.search(query_terms, engine=args.engine)
    search_time = time.time() - start_time

    print(f"Знайдено документів: {len(results)} (за {search_time:.6f} сек)")
    
    if results:
        # Для простоти сортуємо результати за частотою
        top_results = sorted(results, key=lambda p: p.frequency, reverse=True)[:5]
        print("\nТоп-5 результатів (за частотою):")
        for res in top_results:
            print(f"  Doc ID: {res.doc_id} | Частота: {res.frequency}")

if __name__ == '__main__':
    main()