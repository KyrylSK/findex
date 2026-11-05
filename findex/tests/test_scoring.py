import pytest
from findex.index import InvertedIndex, Posting
from findex.scoring import BM25, TfIdf

@pytest.fixture
def mock_index():
    """Створюємо штучний індекс для перевірки математики алгоритмів."""
    idx = InvertedIndex(store_positions=False)
    # Слово "the" зустрічається у всіх 3 документах (дуже часте)
    # Слово "python" - тільки в одному (рідкісне)
    idx.add_document(0, ["the", "the", "the", "python"])
    idx.add_document(1, ["the", "java"])
    idx.add_document(2, ["the", "web", "development"])
    return idx

def test_bm25_rare_vs_frequent(mock_index):
    """Sanity check: рідкісне слово має отримати вищий бал BM25, ніж дуже часте."""
    scorer = BM25()
    
    p_the = mock_index.postings["the"][0]
    p_python = mock_index.postings["python"][0]
    
    score_the = scorer.score(p_the, mock_index)
    score_python = scorer.score(p_python, mock_index)
    
    assert score_python > score_the

def test_tfidf_zero_division_protection(mock_index):
    """Перевірка захисту від ділення на нуль (якщо слова немає в індексі)."""
    scorer = TfIdf()
    p_fake = Posting(doc_id=99, frequency=1)
    assert scorer.score(p_fake, mock_index) == 0.0

def test_bm25_score_positive(mock_index):
    """Бал існуючого слова має бути більшим за нуль."""
    scorer = BM25()
    p = mock_index.postings["java"][0]
    assert scorer.score(p, mock_index) > 0