import pathlib
import re
import unicodedata

def iter_documents(corpus_path: pathlib.Path):
    # Шукаємо всі .txt файли і видаємо їх текст по черзі (ліниво)
    for file_path in corpus_path.rglob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            yield f.read()

def tokenize(text: str):
    # Нормалізація Unicode та переведення в нижній регістр
    text = unicodedata.normalize('NFKC', text.lower())
    # Регулярний вираз для слів (включаючи апострофи, дефіси та цифри)
    pattern = re.compile(r"\b[\w'-]+\b")
    
    # Видаємо токени (слова) по одному
    for match in pattern.finditer(text):
        yield match.group()