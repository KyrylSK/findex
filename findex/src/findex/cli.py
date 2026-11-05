import sys
import json
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from findex.core import tokenize
from findex.index import InvertedIndex, open_index
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

@app.command()
def index(
    corpus: Path = typer.Argument(..., help="Шлях до папки з корпусом документів"),
    output: Path = typer.Option(Path("index.json"), "--output", "-o", help="Файл для збереження індексу"),
    store_positions: bool = typer.Option(False, "--store-positions", help="Зберігати позиції токенів")
) -> None:
    """Побудова індексу з корпусу файлів."""
    if not corpus.is_dir():
        err_console.print(f"[bold red]Помилка:[/bold red] Папка '{corpus}' не існує.")
        raise typer.Exit(code=1)

    idx = InvertedIndex(store_positions=store_positions)
    files = list(corpus.glob("*.txt"))
    if not files:
        err_console.print(f"[bold red]Помилка:[/bold red] У папці '{corpus}' немає .txt файлів.")
        raise typer.Exit(code=1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=err_console) as progress:
        task = progress.add_task(description="Індексація документів...", total=len(files))
        for doc_id, file_path in enumerate(files):
            try:
                text = file_path.read_text(encoding="utf-8")
                tokens = tokenize(text)
                idx.add_document(doc_id, tokens)
            except Exception as e:
                logging.warning(f"Не вдалося прочитати {file_path}: {e}")
            progress.advance(task)

    if output.suffix == ".json":
        idx.save_json(str(output))
    else:
        idx.save_pickle(str(output))

    console.print(f"[bold green]Успішно![/bold green] Створено індекс: {idx} -> '{output}'")

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
    table.add_column("Score (BM25/TF-IDF)", style="green")

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