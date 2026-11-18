import numpy as np
import pytest

from findex.core import tokenize
from findex.index import InvertedIndex
from findex.scoring import BM25, TfIdf, SearchResult
from findex.search import execute_search, execute_search_scalar
from findex.semantic import SemanticIndex, reciprocal_rank_fusion


@pytest.fixture
def benchmark_index() -> InvertedIndex:
    idx = InvertedIndex(store_positions=False)
    docs = [
        "python async crawler fast search engine index",
        "python numpy vectorization bm25 scoring optimization",
        "web server fastapi jinja2 html templates routing",
        "semantic search embeddings cosine similarity transformer",
        "python performance profiling cprofile flamegraph benchmark",
        "information retrieval inverted index postings compression",
    ]
    for doc_id, text in enumerate(docs * 20):
        idx.add_document(doc_id, tokenize(text))
    idx.finalize()
    return idx


def test_numpy_storage_dtypes(benchmark_index: InvertedIndex) -> None:
    """Перевірка, що постінги та довжини документів зберігаються у np.int32."""
    assert benchmark_index.doc_lengths.dtype == np.int32
    doc_ids, freqs = benchmark_index.get_term_arrays("python")
    assert doc_ids.dtype == np.int32
    assert freqs.dtype == np.int32
    assert len(doc_ids) == len(freqs) > 0


@pytest.mark.parametrize("scorer_cls", [BM25, TfIdf])
def test_numpy_order_matches_lab3_scalar(benchmark_index: InvertedIndex, scorer_cls) -> None:
    """Порядок результатів та бали NumPy-скорера збігаються зі скалярним скорером з лаби 3."""
    scorer = scorer_cls()
    query = "python OR search OR index"

    res_scalar = execute_search_scalar(query, benchmark_index, scorer, top_k=10)
    res_numpy = execute_search(query, benchmark_index, scorer, top_k=10)

    assert [r.doc_id for r in res_numpy] == [r.doc_id for r in res_scalar]
    for r_np, r_sc in zip(res_numpy, res_scalar):
        assert r_np.score == pytest.approx(r_sc.score, rel=1e-9)


def test_semantic_best_chunk_and_rrf() -> None:
    """Перевірка косинусної близькості по найкращому фрагменту та злиття через RRF."""
    # 3 фрагменти для 2 документів (doc 0 має 2 фрагменти, doc 1 має 1 фрагмент)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],  # doc 0, chunk 0
            [0.0, 1.0, 0.0],  # doc 0, chunk 1 (найкращий збіг із запитом)
            [0.6, 0.8, 0.0],  # doc 1, chunk 0
        ],
        dtype=np.float32,
    )
    sem_idx = SemanticIndex(
        embeddings=embeddings,
        chunk_doc_ids=np.array([0, 0, 1], dtype=np.int32),
        chunks_text=["chunk a", "chunk b", "chunk c"],
        doc_paths={0: "doc0.txt", 1: "doc1.txt"},
    )
    q_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sem_results = sem_idx.search("ignored", top_k=2, query_vector=q_vec)

    assert len(sem_results) == 2
    assert sem_results[0].doc_id == 0
    assert sem_results[0].score == pytest.approx(1.0)
    assert sem_results[1].doc_id == 1
    assert sem_results[1].score == pytest.approx(0.8)

    lex_results = [SearchResult(score=2.5, doc_id=1), SearchResult(score=1.2, doc_id=0)]
    hybrid = reciprocal_rank_fusion(lex_results, sem_results, top_k=2, rrf_k=60)
    assert len(hybrid) == 2
    assert {r.doc_id for r in hybrid} == {0, 1}


def test_benchmark_search(benchmark, benchmark_index: InvertedIndex) -> None:
    """pytest-benchmark для search (пошук за трьома термами)."""
    scorer = BM25()
    results = benchmark(execute_search, "python OR search OR index", benchmark_index, scorer, 5)
    assert len(results) == 5


def test_benchmark_build_index(benchmark) -> None:
    """pytest-benchmark для build_index (побудова індексу з токенів)."""
    tokenized_docs = [
        (i, tokenize("python async crawler fast search engine index bm25 numpy"))
        for i in range(100)
    ]

    def _build() -> InvertedIndex:
        idx = InvertedIndex(store_positions=False)
        idx.build_from_pipeline(tokenized_docs)
        return idx

    built = benchmark(_build)
    assert built.num_docs() == 100