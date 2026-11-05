import pytest
import tempfile
from pathlib import Path
from hypothesis import given, settings, strategies as st
from findex.index import InvertedIndex
from findex.query import Term, And, Or

# Стратегія для генерації випадкових слів (лише латиниця)
word_strategy = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10)

@given(st.lists(word_strategy, min_size=1, max_size=20))
@settings(max_examples=50)
def test_postings_are_sorted(words):
    """Властивість 1: Постінги завжди відсортовані за doc_id."""
    idx = InvertedIndex()
    idx.add_document(1, words[:len(words)//2])
    idx.add_document(2, words[len(words)//2:])
    
    for term in idx:
        doc_ids = [p.doc_id for p in idx.postings[term]]
        assert doc_ids == sorted(doc_ids)

@given(word_strategy, word_strategy)
@settings(max_examples=50)
def test_merge_coincides_with_set(w1, w2):
    """Властивість 2: Операції парсера (merge) збігаються з множинами (set) Python."""
    idx = InvertedIndex()
    idx.add_document(1, [w1, w2])
    idx.add_document(2, [w1])
    idx.add_document(3, [w2])
    
    set_w1 = Term(w1).evaluate(idx)
    set_w2 = Term(w2).evaluate(idx)
    
    assert And(Term(w1), Term(w2)).evaluate(idx) == set_w1.intersection(set_w2)
    assert Or(Term(w1), Term(w2)).evaluate(idx) == set_w1.union(set_w2)

@given(st.lists(word_strategy, min_size=1, max_size=10))
@settings(max_examples=20)
@pytest.mark.slow
def test_save_load_json_property(words):
    """Властивість 3: load(save(index)) дає той самий індекс (ідемпотентність)."""
    idx = InvertedIndex()
    idx.add_document(1, words)
    
    # Використовуємо tempfile замість tmp_path для сумісності з Hypothesis
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "test_idx.json"
        idx.save_json(str(filepath))
        
        loaded_idx = InvertedIndex.load_json(str(filepath))
        
    assert idx.num_docs() == loaded_idx.num_docs()
    assert set(idx.postings.keys()) == set(loaded_idx.postings.keys())