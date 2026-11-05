import pytest
from findex.core import tokenize

@pytest.mark.parametrize("text, expected", [
    ("Hello World!", ["hello", "world"]),
    ("Python 3.10 is fast", ["python", "3", "10", "is", "fast"]),
    ("Юнікод і кирилиця працюють", ["юнікод", "і", "кирилиця", "працюють"]), # Перевірка Unicode
    ("   Spaces   and\tTabs\n", ["spaces", "and", "tabs"]),
    ("", []),
    ("!@#$%^&*()", [])
])
def test_tokenize(text: str, expected: list[str]) -> None:
    """Перевірка токенізатора з різними вхідними даними (parametrize + Unicode)."""
    assert tokenize(text) == expected