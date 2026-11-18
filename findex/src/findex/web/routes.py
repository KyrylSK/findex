import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal, List
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader

from findex.index import open_index
from findex.scoring import BM25, TfIdf, Scorer, SearchResult
from findex.search import execute_search
from findex.semantic import SemanticIndex, reciprocal_rank_fusion
from findex.web.config import settings
from findex.web.state import app_state

logger = logging.getLogger("findex.web")


def _get_doc_meta(index, doc_id: int):
    doc_meta = getattr(index, "doc_meta", None)
    if doc_meta is None:
        return None
    if isinstance(doc_meta, dict):
        if doc_id in doc_meta:
            return doc_meta[doc_id]
        if str(doc_id) in doc_meta:
            return doc_meta[str(doc_id)]
        return None
    if isinstance(doc_meta, (list, tuple)) and 0 <= doc_id < len(doc_meta):
        return doc_meta[doc_id]
    return None


def _extract_doc_path(meta) -> str:
    if isinstance(meta, dict):
        return str(meta.get("path") or meta.get("url") or meta.get("file") or meta)
    if hasattr(meta, "path"):
        return str(meta.path)
    if hasattr(meta, "url"):
        return str(meta.url)
    return str(meta)


def _run_lexical_web(query: str, index, scorer: Scorer, top_k: int) -> List[SearchResult]:
    try:
        return execute_search(query, index, scorer, top_k=top_k)
    except ValueError:
        words = [w.strip() for w in query.split() if w.strip()]
        if len(words) > 1:
            return execute_search(" OR ".join(words), index, scorer, top_k=top_k)
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.index_path.exists():
        try:
            with open_index(str(settings.index_path)) as idx:
                app_state.index = idx
                logger.info(f"Індекс успішно завантажено з {settings.index_path}")
        except Exception as e:
            logger.error(f"Помилка завантаження індексу: {e}")
    if settings.embeddings_path.exists():
        try:
            app_state.semantic_index = SemanticIndex.load(settings.embeddings_path)
            logger.info(f"Ембединги успішно завантажено з {settings.embeddings_path}")
        except Exception as e:
            logger.error(f"Помилка завантаження ембедингів: {e}")
    yield
    app_state.index = None
    app_state.semantic_index = None


app = FastAPI(title="findex Web Search", version="1.0.0", lifespan=lifespan)

templates_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import uuid
    err_id = uuid.uuid4()
    logger.error(f"Request ID: {err_id} - Unexpected error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_id": str(err_id)},
    )


@app.get("/health")
def health_check():
    if app_state.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return {"status": "ok"}
    

@app.get("/stats")
def stats():
    if app_state.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    terms_dict = getattr(app_state.index, "index", getattr(app_state.index, "postings", {}))
    doc_meta = getattr(app_state.index, "doc_meta", {})
    chunks_count = (
        int(len(app_state.semantic_index.chunk_doc_ids))
        if app_state.semantic_index is not None
        else 0
    )
    return {
        "documents_count": len(doc_meta),
        "terms_count": len(terms_dict),
        "semantic_chunks_count": chunks_count,
    }


def _execute_mode_search(q: str, mode: str, scorer: str, top_k: int) -> List[SearchResult]:
    scoring_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()
    if mode == "lexical":
        if app_state.index is None:
            raise HTTPException(status_code=503, detail="Index not loaded")
        return _run_lexical_web(q, app_state.index, scoring_obj, top_k=top_k)
    if mode == "semantic":
        if app_state.semantic_index is None:
            raise HTTPException(status_code=503, detail="Semantic embeddings not loaded")
        return app_state.semantic_index.search(q, top_k=top_k)
    if app_state.index is None or app_state.semantic_index is None:
        raise HTTPException(status_code=503, detail="Both lexical index and semantic embeddings are required for hybrid mode")
    candidate_k = max(top_k * 3, 20)
    lex_res = _run_lexical_web(q, app_state.index, scoring_obj, top_k=candidate_k)
    sem_res = app_state.semantic_index.search(q, top_k=candidate_k)
    return reciprocal_rank_fusion(lex_res, sem_res, top_k=top_k)


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Пошуковий запит"),
    k: int = Query(5, ge=1, le=50, description="Кількість результатів"),
    scorer: Literal["bm25", "tfidf"] = Query("bm25", description="Алгоритм ранжування"),
    mode: Literal["lexical", "semantic", "hybrid"] = Query("lexical", description="Режим пошуку"),
    page: int = Query(1, ge=1, description="Номер сторінки"),
):
    results = _execute_mode_search(q=q, mode=mode, scorer=scorer, top_k=k * page)

    start_idx = (page - 1) * k
    end_idx = start_idx + k
    page_results = results[start_idx:end_idx]

    return {
        "query": q,
        "mode": mode,
        "scorer": scorer,
        "page": page,
        "k": k,
        "total_results": len(results),
        "results": [{"doc_id": r.doc_id, "score": r.score} for r in page_results],
    }


@app.get("/docs/{doc_id}")
def get_document(doc_id: int):
    if app_state.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    meta = _get_doc_meta(app_state.index, doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_path = _extract_doc_path(meta)
    try:
        content = Path(doc_path).read_text(encoding="utf-8")
    except Exception:
        content = "Не вдалося прочитати файл документа."

    return {
        "doc_id": doc_id,
        "path": doc_path,
        "content": content,
    }


@app.get("/", response_class=HTMLResponse)
def index_page(
    request: Request,
    q: Optional[str] = None,
    scorer: str = "bm25",
    mode: str = "lexical",
    page: int = 1,
):
    results = []
    error_msg = None
    if q:
        try:
            raw_res = _execute_mode_search(q=q, mode=mode, scorer=scorer, top_k=20)
            for r in raw_res:
                doc_path = ""
                if app_state.semantic_index and r.doc_id in app_state.semantic_index.doc_paths:
                    doc_path = app_state.semantic_index.doc_paths[r.doc_id]
                elif app_state.index is not None:
                    meta = _get_doc_meta(app_state.index, r.doc_id)
                    doc_path = _extract_doc_path(meta) if meta is not None else ""
                try:
                    snippet = Path(doc_path).read_text(encoding="utf-8")[:200] + "..."
                except Exception:
                    snippet = "..."
                results.append(
                    {
                        "doc_id": int(r.doc_id),
                        "score": float(r.score),
                        "snippet": str(snippet),
                        "path": doc_path,
                    }
                )
        except HTTPException as he:
            error_msg = he.detail
        except Exception as e:
            logger.error(f"Search error on web page: {e}")
            error_msg = str(e)

    template = templates_env.get_template("index.html")
    html_content = template.render(
        request=request,
        q=q or "",
        scorer=scorer,
        mode=mode,
        page=page,
        results=results,
        error_msg=error_msg,
    )
    return HTMLResponse(content=html_content)