import sys
import json
import logging
import time
import os
from pathlib import Path
from typing import Optional, List, Tuple
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import concurrent.futures
from functools import partial

# Спроба імпортувати psutil для вимірювання пам'яті
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from findex.core import tokenize
from findex.index import InvertedIndex, open_index, build_partial
from findex.scoring import BM25, TfIdf, Scorer
from findex.search import execute_search
from findex.stats import get_corpus_stats

app = typer.Typer(help="findex: Пошуковий рушій та інструментарій CLI", add_completion=False)
console = Console()
err_console = Console(stderr=True)

def setup_logging(verbosity: int) -> None:
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s", stream=sys.stderr)

@app.callback()
def main(
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Рівень деталізації логів (-v або -vv)")
) -> None:
    setup_logging(verbose)

def chunk_data(data: list, num_chunks: int) -> list:
    """Розбиває список на приблизно рівні частини для воркерів."""
    if num_chunks <= 0:
        return [data]
    chunk_size = max(1, len(data) // num_chunks)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    # Якщо залишився хвіст (через ділення з остачею), прикріплюємо його до останнього чанка
    if len(chunks) > num_chunks:
        tail = chunks.pop()
        chunks[-1].extend(tail)
    return chunks

@app.command()
def index(
    corpus: Path = typer.Argument(..., help="Шлях до папки з корпусом документів"),
    output: Path = typer.Option(Path("index.json"), "--output", "-o", help="Файл для збереження індексу"),
    store_positions: bool = typer.Option(False, "--store-positions", help="Зберігати позиції токенів"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість воркерів"),
    executor: str = typer.Option("serial", "--executor", "-e", help="Тип виконання: serial, threads, processes")
) -> None:
    """Побудова індексу з корпусу файлів із підтримкою паралелізму."""
    if not corpus.is_dir():
        err_console.print(f"[bold red]Помилка:[/bold red] Папка '{corpus}' не існує.")
        raise typer.Exit(code=1)

    files = list(corpus.glob("*.txt"))
    if not files:
        err_console.print(f"[bold red]Помилка:[/bold red] У папці '{corpus}' немає .txt файлів.")
        raise typer.Exit(code=1)

    docs_info: List[Tuple[int, Path]] = [(i, f) for i, f in enumerate(files)]
    num_chunks = workers if executor != "serial" else 1
    chunks = chunk_data(docs_info, num_chunks)

    console.print(f"Починаємо індексацію: {len(files)} файлів, {workers} воркер(ів), {executor}.")
    
    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    partial_indices = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=err_console) as progress:
        task = progress.add_task(description=f"Індексація ({executor})...", total=len(chunks))
        
        # Використовуємо partial, щоб передати параметр store_positions
        func = partial(build_partial, store_positions=store_positions)
        
        if executor == "serial":
            for chunk in chunks:
                partial_indices.append(func(chunk))
                progress.advance(task)
                
        elif executor == "threads":
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(func, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    partial_indices.append(future.result())
                    progress.advance(task)
                    
        elif executor == "processes":
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(func, chunk) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    partial_indices.append(future.result())
                    progress.advance(task)
        else:
            err_console.print(f"[bold red]Помилка:[/bold red] Невідомий executor '{executor}'.")
            raise typer.Exit(code=1)

    # Фаза злиття (merge)
    merge_start = time.perf_counter()
    main_index = InvertedIndex(store_positions=store_positions)
    for p_idx in partial_indices:
        main_index.merge(p_idx)
    merge_end = time.perf_counter()

    end_wall = time.perf_counter()
    end_cpu = time.process_time()

    if output.suffix == ".json":
        main_index.save_json(str(output))
    else:
        main_index.save_pickle(str(output))

    console.print(f"[bold green]Успішно![/bold green] Створено індекс: {main_index} -> '{output}'")
    
    # Вивід статистики для бенчмаркінгу
    wall_time = end_wall - start_wall
    cpu_time = end_cpu - start_cpu
    merge_time = merge_end - merge_start
    
    stat_table = Table(title="Метрики продуктивності")
    stat_table.add_column("Метрика", style="cyan")
    stat_table.add_column("Значення", style="magenta")
    stat_table.add_row("Wall time (загальний)", f"{wall_time:.4f} с")
    stat_table.add_row("CPU time (батьківський процес)", f"{cpu_time:.4f} с")
    stat_table.add_row("Merge time (злиття)", f"{merge_time:.4f} с")
    
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)
        stat_table.add_row("RSS (пік пам'яті батька)", f"{rss_mb:.2f} MB")
    else:
        stat_table.add_row("RSS (пік пам'яті батька)", "psutil не встановлено")
        
    console.print(stat_table)

@app.command()
def search(
    query: str = typer.Argument(..., help="Пошуковий запит"),
    index_file: Path = typer.Option(Path("index.json"), "--index", "-i", help="Шлях до файлу індексу"),
    scorer_type: str = typer.Option("bm25", "--scorer", "-s", help="Алгоритм: bm25 або tfidf"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Кількість результатів"),
    json_output: bool = typer.Option(False, "--json", help="Вивід у форматі JSON в stdout")
) -> None:
    """Пошук у побудованому індексі."""
    if not index_file.exists():
        err_console.print(f"[bold red]Помилка:[/bold red] Файл індексу '{index_file}' не знайдено.")
        raise typer.Exit(code=1)

    scorer: Scorer = BM25() if scorer_type.lower() == "bm25" else TfIdf()

    try:
        with open_index(str(index_file)) as idx:
            results = execute_search(query, idx, scorer, top_k=top_k)
    except Exception as e:
        err_console.print(f"[bold red]Помилка запиту:[/bold red] {e}")
        raise typer.Exit(code=1)

    if json_output:
        data = [{"doc_id": r.doc_id, "score": round(r.score, 4)} for r in results]
        print(json.dumps(data, indent=2))
        return

    if not results:
        console.print("[yellow]Нічого не знайдено.[/yellow]")
        return

    table = Table(title=f"Результати пошуку для: '{query}'")
    table.add_column("№", style="cyan", justify="right")
    table.add_column("Doc ID", style="magenta")
    table.add_column("Score", style="green")

    for i, res in enumerate(results, 1):
        table.add_row(str(i), str(res.doc_id), f"{res.score:.4f}")

    console.print(table)

@app.command()
def stats(
    corpus: Path = typer.Argument(..., help="Шлях до папки з корпусом документів")
) -> None:
    """Статистика корпусу текстів."""
    if not corpus.is_dir():
        err_console.print(f"[bold red]Помилка:[/bold red] Папка '{corpus}' не існує.")
        raise typer.Exit(code=1)

    total_words, unique_words, top_10 = get_corpus_stats(corpus)

    table = Table(title=f"Статистика корпусу '{corpus}'")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значення", style="green")

    table.add_row("Усього слів", str(total_words))
    table.add_row("Унікальних слів (словник)", str(unique_words))

    console.print(table)

    top_table = Table(title="Топ-10 найчастіших слів")
    top_table.add_column("Слово", style="yellow")
    top_table.add_column("Частота", style="magenta", justify="right")

    for word, count in top_10:
        top_table.add_row(word, str(count))

    console.print(top_table)

if __name__ == "__main__":
    app()