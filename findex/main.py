import argparse
from pathlib import Path

# Импортируем всё, что мы написали
from findex.index import open_index
from findex.search import execute_search, build_snippet
from findex.query import parse_query
# Импортируем твою функцию чтения оригинальных текстов (подставь свой правильный путь, если он другой)
from findex.core import iter_documents 

def main():
    parser = argparse.ArgumentParser(description="Поиск с ранжированием и сниппетами")
    parser.add_argument("--index", required=True, help="Путь к файлу индекса (.json или .bin)")
    parser.add_argument("--data", required=True, help="Путь к папке с исходными текстами (для сниппетов)")
    args = parser.parse_args()

    print(f"Загрузка индекса из {args.index}...")
    
    # 1. Используем наш Context Manager для безопасного открытия
    with open_index(args.index) as index:
        print(index) # Вызовет наш __repr__
        print(f"Средняя длина документа: {index.avg_doc_length:.2f} слов\n")
        
        # 2. Загружаем оригинальные тексты в память, чтобы вырезать из них сниппеты
        print(f"Загрузка текстов из {args.data}...")
        original_texts = {}
        for doc_id, text in enumerate(iter_documents(Path(args.data))):
            original_texts[doc_id] = text
            
        print("Готово. Введите запрос (или 'exit' для выхода).")
        print("Поддерживаются: AND, OR, NOT, (дужки), \"точные фразы\"")
        print("-" * 50)
        
        # 3. Интерактивный цикл поиска
        while True:
            try:
                query_str = input("\nПоиск> ").strip()
                if query_str.lower() in ('exit', 'quit'):
                    break
                if not query_str:
                    continue
                    
                # Выполняем поиск (здесь сработает декоратор @timed)
                results = execute_search(query_str, index, top_k=5)
                
                if not results:
                    print("Ничего не найдено.")
                    continue
                    
                # Извлекаем слова запроса для подсветки
                ast = parse_query(query_str)
                # Вспомогательная функция для извлечения слов (просто дублируем логику из search.py)
                def get_words(node):
                    from findex.query import Term, Phrase
                    w = set()
                    if isinstance(node, Term): w.add(node.word)
                    elif isinstance(node, Phrase): w.update(node.words)
                    elif hasattr(node, 'left'): 
                        w.update(get_words(node.left))
                        w.update(get_words(node.right))
                    elif hasattr(node, 'child'): w.update(get_words(node.child))
                    return w
                    
                query_words = get_words(ast)
                
                # 4. Вывод результатов
                print(f"\nТоп-{len(results)} результатов:")
                for i, res in enumerate(results, 1):
                    print(f"\n[{i}] Документ ID: {res.doc_id} (Бал BM25: {res.score:.4f})")
                    
                    # Генерируем сниппет
                    text = original_texts.get(res.doc_id, "")
                    snippet = build_snippet(text, query_words)
                    print(f"Сниппет: {snippet}")
                    
            except Exception as e:
                print(f"Ошибка запроса: {e}")

if __name__ == '__main__':
    main()