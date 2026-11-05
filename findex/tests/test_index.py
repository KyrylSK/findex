import pytest
from findex.index import InvertedIndex, open_index

@pytest.fixture
def sample_index():
    idx = InvertedIndex(store_positions=False)
    idx.add_document(1, ["hello", "world"])
    idx.add_document(2, ["hello", "python"])
    return idx

def test_index_len(sample_index):
    assert len(sample_index) == 3

def test_index_contains(sample_index):
    assert "hello" in sample_index
    assert "java" not in sample_index

def test_index_iter(sample_index):
    terms = list(sample_index)
    assert "hello" in terms

def test_index_repr(sample_index):
    assert "<InvertedIndex" in repr(sample_index)

def test_index_num_docs(sample_index):
    assert sample_index.num_docs() == 2

def test_index_df(sample_index):
    assert sample_index.df("hello") == 2
    assert sample_index.df("python") == 1

def test_index_avg_doc_length(sample_index):
    assert sample_index.avg_doc_length == 2.0

def test_save_load_pickle(sample_index, tmp_path):
    path = tmp_path / "index.bin"
    sample_index.save_pickle(str(path))
    loaded = InvertedIndex.load_pickle(str(path))
    assert len(loaded) == len(sample_index)

def test_open_index_context_json(sample_index, tmp_path):
    path = tmp_path / "index.json"
    sample_index.save_json(str(path))
    with open_index(str(path)) as idx:
        assert "hello" in idx

def test_open_index_context_pickle(sample_index, tmp_path):
    path = tmp_path / "index.bin"
    sample_index.save_pickle(str(path))
    with open_index(str(path)) as idx:
        assert "hello" in idx