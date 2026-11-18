import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from findex.scoring import SearchResult

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """Ділить текст документа на фрагменти (чанки) із перекриттям слів."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass
class SemanticIndex:
    """Семантичний індекс: нормалізована матриця ембедингів (.npy) + мапінг фрагментів на doc_id."""
    embeddings: np.ndarray
    chunk_doc_ids: np.ndarray
    chunks_text: List[str]
    doc_paths: Dict[int, str]
    model_name: str = DEFAULT_MODEL_NAME
    _model: Optional[object] = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """Кодує запит у L2-нормалізований вектор float32."""
        model = self._get_model()
        vec = model.encode([query], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: int = 5,
        query_vector: Optional[np.ndarray] = None,
    ) -> List[SearchResult]:
        """Множить матрицю ембедингів на вектор запиту та зараховує документу його найкращий фрагмент."""
        if self.embeddings.size == 0 or len(self.chunk_doc_ids) == 0 or top_k <= 0:
            return []

        q_vec = query_vector if query_vector is not None else self.encode_query(query)
        # Косинусна близькість (оскільки всі вектори L2-нормалізовані)
        chunk_scores: np.ndarray = self.embeddings @ q_vec

        # Агрегація по doc_id: кожному документу зараховується максимальний бал серед його фрагментів
        best_doc_scores: Dict[int, float] = {}
        for doc_id, score in zip(self.chunk_doc_ids.tolist(), chunk_scores.tolist()):
            prev = best_doc_scores.get(doc_id)
            if prev is None or score > prev:
                best_doc_scores[doc_id] = float(score)

        if not best_doc_scores:
            return []

        doc_ids_arr = np.fromiter(best_doc_scores.keys(), dtype=np.int32, count=len(best_doc_scores))
        scores_arr = np.fromiter(best_doc_scores.values(), dtype=np.float64, count=len(best_doc_scores))

        n = len(scores_arr)
        if n <= top_k:
            top_idx = np.argsort(-scores_arr)
        else:
            part_idx = np.argpartition(-scores_arr, top_k - 1)[:top_k]
            top_idx = part_idx[np.argsort(-scores_arr[part_idx])]

        return [
            SearchResult(score=float(scores_arr[i]), doc_id=int(doc_ids_arr[i]))
            for i in top_idx
        ]

    def save(self, npy_path: str | Path) -> Tuple[Path, Path]:
        """Зберігає матрицю ембедингів у .npy та метадані у .meta.json."""
        out_npy = Path(npy_path)
        out_meta = out_npy.with_suffix(".meta.json")
        np.save(out_npy, self.embeddings.astype(np.float32, copy=False))
        meta = {
            "model_name": self.model_name,
            "chunk_doc_ids": self.chunk_doc_ids.tolist(),
            "chunks_text": self.chunks_text,
            "doc_paths": {str(k): v for k, v in self.doc_paths.items()},
        }
        out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_npy, out_meta

    @classmethod
    def load(cls, npy_path: str | Path) -> "SemanticIndex":
        """Завантажує матрицю ембедингів з .npy та метадані з .meta.json."""
        in_npy = Path(npy_path)
        in_meta = in_npy.with_suffix(".meta.json")
        embeddings = np.load(in_npy).astype(np.float32, copy=False)
        meta = json.loads(in_meta.read_text(encoding="utf-8"))
        chunk_doc_ids = np.asarray(meta["chunk_doc_ids"], dtype=np.int32)
        chunks_text = list(meta.get("chunks_text", []))
        doc_paths = {int(k): v for k, v in meta.get("doc_paths", {}).items()}
        model_name = str(meta.get("model_name", DEFAULT_MODEL_NAME))
        return cls(
            embeddings=embeddings,
            chunk_doc_ids=chunk_doc_ids,
            chunks_text=chunks_text,
            doc_paths=doc_paths,
            model_name=model_name,
        )


def build_semantic_index(
    corpus_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    chunk_size: int = 120,
    overlap: int = 30,
) -> SemanticIndex:
    """Будує семантичний індекс для всіх .txt файлів у папці корпусу."""
    from sentence_transformers import SentenceTransformer

    corpus = Path(corpus_dir)
    files = list(corpus.glob("*.txt"))

    all_chunks: List[str] = []
    chunk_doc_ids: List[int] = []
    doc_paths: Dict[int, str] = {}

    for doc_id, file_path in enumerate(files):
        doc_paths[doc_id] = str(file_path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for ch in chunks:
            all_chunks.append(ch)
            chunk_doc_ids.append(doc_id)

    model = SentenceTransformer(model_name)
    if all_chunks:
        embeddings = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=False)
        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
    else:
        dim = model.get_sentence_embedding_dimension() or 384
        embeddings_arr = np.zeros((0, dim), dtype=np.float32)

    return SemanticIndex(
        embeddings=embeddings_arr,
        chunk_doc_ids=np.asarray(chunk_doc_ids, dtype=np.int32),
        chunks_text=all_chunks,
        doc_paths=doc_paths,
        model_name=model_name,
        _model=model,
    )


def reciprocal_rank_fusion(
    lexical_results: List[SearchResult],
    semantic_results: List[SearchResult],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[SearchResult]:
    """Злиття лексичного (BM25) та семантичного рейтингів через Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[int, float] = {}

    for rank, res in enumerate(lexical_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, res in enumerate(semantic_results, start=1):
        rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + 1.0 / (rrf_k + rank)

    if not rrf_scores or top_k <= 0:
        return []

    fused = [SearchResult(score=score, doc_id=doc_id) for doc_id, score in rrf_scores.items()]
    fused.sort()
    return fused[:top_k]