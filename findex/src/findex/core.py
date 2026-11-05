import re
from pathlib import Path
from typing import Iterator

def tokenize(text: str) -> list[str]:
    """Розбиває текст на слова (токени), видаляє пунктуацію та переводить у нижній регістр."""
    # Знаходимо всі слова, які складаються з літер або цифр
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def iter_documents(corpus_path: Path) -> Iterator[str]:
    """
    Ліниво (по одному) читає всі .txt файли з папки corpus_path.
    Повертає вміст кожного файлу як рядок.
    """
    if not corpus_path.is_dir():
        raise ValueError(f"Шлях {corpus_path} не є папкою")

    for file_path in corpus_path.glob("*.txt"):
        if file_path.is_file():
            # Читаємо файл і повертаємо його вміст (yield робить функцію генератором)
            with open(file_path, 'r', encoding='utf-8') as f:
                yield f.read()