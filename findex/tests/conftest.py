import pytest
from pathlib import Path
from typing import Generator

@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """
    Фікстура, що створює тимчасовий міні-корпус текстів для тестів.
    tmp_path - це вбудована фікстура pytest для безпечної роботи з тимчасовими файлами.
    """
    docs = {
        "doc0.txt": "Python is great for web development.",
        "doc1.txt": "Java is heavily used in enterprise. Python is too.",
        "doc2.txt": "Web development is fun."
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    
    return tmp_path