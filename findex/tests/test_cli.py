from pathlib import Path
from typer.testing import CliRunner
from findex.cli import app

runner = CliRunner()

def test_cli_help():
    """Перевірка виводу довідки."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "findex:" in result.stdout

def test_cli_index_and_search(sample_corpus: Path, tmp_path: Path):
    """Повний цикл: індексація та пошук через CLI."""
    index_file = tmp_path / "idx.json"
    
    # 1. Створення індексу
    res_idx = runner.invoke(app, ["index", str(sample_corpus), "-o", str(index_file)])
    assert res_idx.exit_code == 0
    assert index_file.exists()

    # 2. Пошук у форматах Table та JSON
    res_search = runner.invoke(app, ["search", "python", "-i", str(index_file)])
    assert res_search.exit_code == 0
    assert "Doc ID" in res_search.stdout

    res_json = runner.invoke(app, ["search", "python", "-i", str(index_file), "--json"])
    assert res_json.exit_code == 0
    assert "[" in res_json.stdout

def test_cli_stats(sample_corpus: Path):
    """Перевірка команди stats."""
    result = runner.invoke(app, ["stats", str(sample_corpus)])
    assert result.exit_code == 0
    assert "Усього слів" in result.stdout

def test_cli_missing_file_error():
    """Перевірка зрозумілої помилки та ненульового коду виходу без traceback."""
    result = runner.invoke(app, ["search", "python", "-i", "non_existent_file.json"])
    assert result.exit_code == 1
    assert "Помилка" in result.stderr