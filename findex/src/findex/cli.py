import sys
import json
import logging
import time
import os
import asyncio
from pathlib import Path
from typing import Optional, List, Tuple
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
import concurrent.futures
from functools import partial

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
    if num_chunks <= 0: return [data]
    chunk_size = max(1, len(data) // num_chunks)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
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

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    partial_indices = []

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=err_console) as progress:
        task = progress.add_task(description=f"Індексація ({executor})...", total=len(chunks))
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
            raise typer.Exit(code=1)

    main_index = InvertedIndex(store_positions=store_positions)
    for p_idx in partial_indices:
        main_index.merge(p_idx)

    if output.suffix == ".json":
        main_index.save_json(str(output))
    else:
        main_index.save_pickle(str(output))

    console.print(f"[bold green]Успішно![/bold green] Створено індекс: {main_index} -> '{output}'")

@app.command()
def search(
    query: str = typer.Argument(..., help="Пошуковий запит"),
    index_file: Path = typer.Option(Path("index.json"), "--index", "-i", help="Шлях до файлу індексу"),
    scorer_type: str = typer.Option("bm25", "--scorer", "-s", help="Алгоритм: bm25 або tfidf"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Кількість результатів"),
    json_output: bool = typer.Option(False, "--json", help="Вивід у форматі JSON в stdout")
) -> None:
    """Пошук у побудованому індексі."""
    scorer: Scorer = BM25() if scorer_type.lower() == "bm25" else TfIdf()
    with open_index(str(index_file)) as idx:
        results = execute_search(query, idx, scorer, top_k=top_k)

    table = Table(title=f"Результати пошуку: '{query}'")
    table.add_column("№", style="cyan", justify="right")
    table.add_column("Doc ID", style="magenta")
    table.add_column("Score", style="green")
    for i, res in enumerate(results, 1):
        table.add_row(str(i), str(res.doc_id), f"{res.score:.4f}")
    console.print(table)

@app.command()
def crawl(
    start_url: str = typer.Argument(..., help="Початковий URL для обходу"),
    max_pages: int = typer.Option(200, "--max-pages", "-m", help="Максимум сторінок"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Кількість воркерів"),
    output: Path = typer.Option(Path("crawled.jsonl"), "--output", "-o", help="Файл JSONL")
) -> None:
    """Асинхронний краулер веб-сторінок."""
    asyncio.run(_async_crawl(start_url, max_pages, concurrency, output))

async def _async_crawl(start_url: str, max_pages: int, concurrency: int, output: Path):
    from findex.crawler import crawl as crawler_gen
    
    # Налаштовуємо запис у crawl.log
    file_handler = logging.FileHandler("crawl.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    crawler_logger = logging.getLogger("findex.crawler")
    crawler_logger.setLevel(logging.INFO)
    crawler_logger.addHandler(file_handler)
    crawler_logger.propagate = False
    
    # Очищуємо вихідний JSONL файл
    output.write_text("", encoding="utf-8")

    crawled_count = 0
    error_count = 0
    start_time = time.perf_counter()

    # Дашборд
    table = Table(title=f"Краулінг: {start_url} (Воркерів: {concurrency})")
    table.add_column("Зібрано", justify="right", style="green")
    table.add_column("Стор/сек", justify="right", style="cyan")
    table.add_column("Помилки", justify="right", style="red")
    table.add_column("Черга", justify="right", style="yellow")
    table.add_column("Останній URL", justify="left", style="magenta")

    with Live(table, refresh_per_second=4) as live, open(output, "a", encoding="utf-8") as f_jsonl:
        async for result in crawler_gen(start_url, max_pages, concurrency):
            status = result["status"]
            url = result["url"]
            text = result["text"]
            
            crawler_logger.info(f"[{status}] {url}")
            
            if status == 200:
                crawled_count += 1
                if text:
                    record = {"url": url, "text": text}
                    f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                error_count += 1
                
            elapsed_total = time.perf_counter() - start_time
            pages_per_sec = crawled_count / elapsed_total if elapsed_total > 0 else 0.0
            
            # Оновлюємо таблицю
            table = Table(title=f"Краулінг: {start_url} (Воркерів: {concurrency})")
            table.add_column("Зібрано", justify="right", style="green")
            table.add_column("Стор/сек", justify="right", style="cyan")
            table.add_column("Помилки", justify="right", style="red")
            table.add_column("Черга", justify="right", style="yellow")
            table.add_column("Останній URL", justify="left", style="magenta")
            
            display_url = url if len(url) <= 50 else url[:47] + "..."
            table.add_row(
                str(crawled_count),
                f"{pages_per_sec:.2f}",
                str(error_count),
                str(result["queue_size"]),
                f"[{status}] {display_url}"
            )
            live.update(table)

    elapsed_total = time.perf_counter() - start_time
    stat_table = Table(title="Фінальна статистика")
    stat_table.add_column("Метрика", style="cyan")
    stat_table.add_column("Значення", style="magenta")
    stat_table.add_row("Wall time (загальний)", f"{elapsed_total:.2f} с")
    stat_table.add_row("Сторінок за секунду", f"{(crawled_count / elapsed_total):.2f}")
    stat_table.add_row("Усього помилок", str(error_count))
    
    if HAS_PSUTIL:
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        stat_table.add_row("RSS (пік пам'яті)", f"{rss_mb:.2f} MB")
        
    console.print(stat_table)
    console.print(f"[bold green]Краулінг завершено![/bold green] Дані збережено в {output}")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Порт для вебсерверу"),
    workers: int = typer.Option(1, "--workers", "-w", help="Кількість воркерів uvicorn")
) -> None:
    """Запуск вебсерверу FastAPI через uvicorn."""
    import uvicorn
    console.print(f"[bold green]Запуск findex web на http://127.0.0.1:{port}[/bold green]")
    uvicorn.run("findex.web.routes:app", host="127.0.0.1", port=port, workers=workers, reload=True)

if __name__ == "__main__":
    app()