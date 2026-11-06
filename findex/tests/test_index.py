import pytest
from findex.index import InvertedIndex, open_index, build_partial

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

def test_build_partial_and_merge(tmp_path):
    """Перевірка, що злиття частин дає той самий індекс, що й монолітна побудова."""
    docs = ["Text one", "Text two", "Text three", "Text four"]
    paths = []
    for i, text in enumerate(docs):
        p = tmp_path / f"doc_{i}.txt"
        p.write_text(text, encoding="utf-8")
        paths.append(p)
        
    idx_mono = InvertedIndex()
    idx_mono.add_document(0, ["text", "one"])
    idx_mono.add_document(1, ["text", "two"])
    idx_mono.add_document(2, ["text", "three"])
    idx_mono.add_document(3, ["text", "four"])
    
    chunk1 = [(0, paths[0]), (1, paths[1])]
    chunk2 = [(2, paths[2]), (3, paths[3])]
    
    idx_part1 = build_partial(chunk1, store_positions=False)
    idx_part2 = build_partial(chunk2, store_positions=False)
    
    idx_merged = InvertedIndex()
    idx_merged.merge(idx_part1)
    idx_merged.merge(idx_part2)
    
    assert idx_mono.num_docs() == idx_merged.num_docs()
    assert set(idx_mono.postings.keys()) == set(idx_merged.postings.keys())
    assert idx_mono.postings["text"] == idx_merged.postings["text"]