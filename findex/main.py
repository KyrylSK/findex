from pathlib import Path
from src.findex.stats import measure_lazy, measure_greedy

if __name__ == "__main__":
    # Вказуємо шлях до папки з книгами
    corpus_path = Path("data")
    
    print("--- Лінивий підхід (Генератори) ---")
    lazy_time, lazy_mem, top_words = measure_lazy(corpus_path)
    print(f"Час: {lazy_time:.2f} сек, Пікова пам'ять: {lazy_mem:.2f} MB")
    print(f"Топ-5 слів: {top_words}\n")

    print("--- Жадібний підхід (Списки) ---")
    greedy_time, greedy_mem = measure_greedy(corpus_path)
    print(f"Час: {greedy_time:.2f} сек, Пікова пам'ять: {greedy_mem:.2f} MB")