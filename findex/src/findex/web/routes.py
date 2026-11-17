import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from jinja2 import Environment, FileSystemLoader

from findex.index import open_index
from findex.scoring import BM25, TfIdf, Scorer
from findex.search import execute_search
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.index_path.exists():
        try:
            with open_index(str(settings.index_path)) as idx:
                app_state.index = idx
                logger.info(f"Індекс успішно завантажено з {settings.index_path}")
        except Exception as e:
            logger.error(f"Помилка завантаження індексу: {e}")
    yield
    app_state.index = None


app = FastAPI(title="findex Web Search", lifespan=lifespan)

templates_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import uuid
    err_id = uuid.uuid4()
    logger.error(f"Request ID: {err_id} - Unexpected error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error_id": str(err_id)}
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
    return {
        "documents_count": len(doc_meta),
        "terms_count": len(terms_dict),
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Пошуковий запит"),
    k: int = Query(5, ge=1, le=50, description="Кількість результатів"),
    scorer: Literal["bm25", "tfidf"] = Query("bm25", description="Алгоритм ранжування"),
    page: int = Query(1, ge=1, description="Номер сторінки")
):
    if app_state.index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    scoring_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()
    results = execute_search(q, app_state.index, scoring_obj, top_k=k * page)

    start_idx = (page - 1) * k
    end_idx = start_idx + k
    page_results = results[start_idx:end_idx]

    return {
        "query": q,
        "scorer": scorer,
        "page": page,
        "k": k,
        "total_results": len(results),
        "results": [{"doc_id": r.doc_id, "score": r.score} for r in page_results]
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
        "content": content
    }


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request, q: Optional[str] = None, scorer: str = "bm25", page: int = 1):
    results = []
    if q and app_state.index:
        try:
            scoring_obj: Scorer = BM25() if scorer == "bm25" else TfIdf()
            raw_res = execute_search(q, app_state.index, scoring_obj, top_k=20)
            for r in raw_res:
                meta = _get_doc_meta(app_state.index, r.doc_id)
                doc_path = _extract_doc_path(meta) if meta is not None else ""
                try:
                    snippet = Path(doc_path).read_text(encoding="utf-8")[:200] + "..."
                except Exception:
                    snippet = "..."
                results.append({
                    "doc_id": int(r.doc_id),
                    "score": float(r.score),
                    "snippet": str(snippet),
                    "path": doc_path
                })
        except Exception as e:
            logger.error(f"Search error on web page: {e}")

    template = templates_env.get_template("index.html")
    html_content = template.render(
        request=request,
        q=q or "",
        scorer=scorer,
        page=page,
        results=results
    )
    return HTMLResponse(content=html_content)