import pytest
from findex.query import parse_query, Term, Phrase, And, Or, Not

def test_parse_term():
    """Перевірка звичайного слова."""
    assert parse_query("python") == Term("python")

def test_parse_phrase():
    """Перевірка точної фрази."""
    assert parse_query('"event loop"') == Phrase(["event", "loop"])

def test_parse_and():
    """Перевірка логічного AND."""
    assert parse_query("python AND java") == And(Term("python"), Term("java"))

def test_parse_or():
    """Перевірка логічного OR."""
    assert parse_query("python OR java") == Or(Term("python"), Term("java"))

def test_parse_not():
    """Перевірка логічного NOT."""
    assert parse_query("NOT windows") == Not(Term("windows"))

def test_parse_complex():
    """Перевірка складного запиту з дужками."""
    # python AND (async OR await)
    expected = And(Term("python"), Or(Term("async"), Term("await")))
    assert parse_query("python AND (async OR await)") == expected

def test_parse_error():
    """Перевірка синтаксичної помилки (незакрита дужка)."""
    with pytest.raises(ValueError):
        parse_query("python AND (java")